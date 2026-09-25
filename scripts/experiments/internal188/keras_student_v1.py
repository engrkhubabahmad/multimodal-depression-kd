"""Keras multimodal student, trained on canonical TRAIN-107; TEST untouched."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix
from .student_temporal_v1 import inputs
from .train_student_kd import digest


def score(y,p):
    z=(np.asarray(p)>=.5).astype(int)
    return {'n':len(y),'accuracy':float(accuracy_score(y,z)),
        'macro_f1':float(f1_score(y,z,average='macro',zero_division=0)),
        'depressed_f1':float(f1_score(y,z,pos_label=1,zero_division=0)),
        'auroc':float(roc_auc_score(y,p)),'confusion_matrix':confusion_matrix(y,z,labels=[0,1]).tolist(),
        'predicted_positive':int(z.sum())}


def targets(exp,ids,y):
    t=exp/'teachers/idiap_text_inductive_v1/train_targets.csv'
    a=exp/'teachers/nusd_ecapa_frozen/evaluation/train_audio_targets.csv'
    df=pd.read_csv(t).merge(pd.read_csv(a),on=['participant_id','label'],validate='one_to_one')
    if len(df)!=107 or df.participant_id.duplicated().any():raise ValueError('Teacher TRAIN coverage mismatch')
    df=df.set_index('participant_id').loc[ids]
    if not np.array_equal(df.label.to_numpy(int),y):raise ValueError('Teacher label mismatch')
    p=df[['text_probability','audio_probability']].to_numpy('float32')
    if not np.isfinite(p).all() or np.any((p<=0)|(p>=1)):raise ValueError('Invalid teacher probabilities')
    selection=json.loads((exp/'teachers/nusd_ecapa_frozen/evaluation/all_five/selection.json').read_text())
    audio_audit=json.loads((a.parent/'train_audio_audit.json').read_text())
    if selection['selected']['run']!=audio_audit['selected_run']:raise ValueError('Audio run mismatch')
    text_audit=json.loads((t.parent/'audit.json').read_text())
    if text_audit['split_sha256']!=digest(exp/'split/manifest.csv'):raise ValueError('Text target split mismatch')
    return p,{'text_path':str(t),'audio_path':str(a),'text_sha256':digest(t),
              'audio_sha256':digest(a),'selected_audio_run':selection['selected']['run']}


def architecture(tf,text_dim):
    keras=tf.keras;L=keras.layers
    text=L.Input((text_dim,),name='text');audio=L.Input((130,32),name='audio');mask=L.Input((2,),name='mask')
    tx=L.Dense(32,activation='relu',kernel_regularizer=keras.regularizers.l2(.002))(text)
    tx=L.LayerNormalization()(tx);tx=L.Dropout(.25)(tx)
    ax=L.Permute((2,1))(audio)
    ax=L.SeparableConv1D(24,5,padding='same',activation='relu')(ax)
    ax=L.SeparableConv1D(16,3,padding='same',activation='relu')(ax)
    ax=L.GlobalAveragePooling1D()(ax)
    tm=L.Lambda(lambda x:x[:,0:1],name='text_mask')(mask)
    am=L.Lambda(lambda x:x[:,1:2],name='audio_mask')(mask)
    joined=L.Concatenate()([L.Multiply()([tx,tm]),L.Multiply()([ax,am]),mask])
    hidden=L.Dense(16,activation='relu',kernel_regularizer=keras.regularizers.l2(.002))(joined)
    hidden=L.Dropout(.2)(hidden)
    prediction=L.Dense(1,activation='sigmoid',name='depression')(hidden)
    return keras.Model([text,audio,mask],prediction,name='keras_temporal_student')


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',required=True,type=Path)
    parser.add_argument('--seed',type=int,default=103);parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=16);args=parser.parse_args(argv)
    if args.epochs<1 or args.batch_size<1:raise ValueError('Invalid epochs/batch size')
    import tensorflow as tf
    exp=args.experiment;data,preprocessing=inputs(exp,args.seed)
    tr,dev=data['train'],data['val'];teacher,teacher_info=targets(exp,tr['ids'],tr['y'])
    output=exp/'students/canonical_v1/keras_temporal_v1';output.mkdir(parents=True,exist_ok=True)
    class Distilled(tf.keras.Model):
        def __init__(self,network,mode):
            super().__init__();self.network=network;self.mode=mode
            self.loss_tracker=tf.keras.metrics.Mean(name='loss')
            self.acc=tf.keras.metrics.BinaryAccuracy(name='accuracy')
        @property
        def metrics(self):return [self.loss_tracker,self.acc]
        def call(self,x,training=False):return self.network([x['text'],x['audio'],x['mask']],training=training)
        def train_step(self,data):
            x,y=data;batch=tf.shape(y)[0]
            draw=tf.random.uniform((batch,))
            mask=tf.stack([tf.cast(draw>=.25,tf.float32),
                           tf.cast(tf.logical_or(draw<.25,draw>=.5),tf.float32)],axis=1)
            noisy={'text':x['text']+tf.random.normal(tf.shape(x['text']),stddev=.05),
                   'audio':x['audio']+tf.random.normal(tf.shape(x['audio']),stddev=.05),'mask':mask}
            with tf.GradientTape() as tape:
                p=tf.clip_by_value(self(noisy,training=True),1e-6,1-1e-6)
                weight=tf.where(tf.squeeze(y,axis=-1)>.5,positive_weight,1.)
                hard=tf.reduce_mean(weight*tf.keras.losses.binary_crossentropy(y,p))
                loss=hard
                if self.mode!='plain':
                    prob=tf.clip_by_value(x['teacher'],1e-6,1-1e-6)
                    if self.mode=='ra_kd':
                        entropy=-(prob*tf.math.log(prob)+(1-prob)*tf.math.log(1-prob))/tf.math.log(2.)
                        reliability=tf.maximum(1-entropy,.05)*mask
                    else:reliability=mask
                    soft=tf.reduce_sum(reliability*prob,axis=1,keepdims=True)/tf.reduce_sum(reliability,axis=1,keepdims=True)
                    kd=tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.stop_gradient(soft),p))
                    loss=.7*hard+.3*kd
                if self.losses:loss+=tf.add_n(self.losses)
            grad=tape.gradient(loss,self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad,self.trainable_variables))
            self.loss_tracker.update_state(loss);self.acc.update_state(y,p)
            return {m.name:m.result() for m in self.metrics}
        def test_step(self,data):
            x,y=data;p=self(x,training=False)
            weight=tf.where(tf.squeeze(y,axis=-1)>.5,positive_weight,1.)
            loss=tf.reduce_mean(weight*tf.keras.losses.binary_crossentropy(y,p))
            if self.losses:loss+=tf.add_n(self.losses)
            self.loss_tracker.update_state(loss);self.acc.update_state(y,p)
            return {m.name:m.result() for m in self.metrics}

    positive_weight=tf.constant((tr['y']==0).sum()/max(1,(tr['y']==1).sum()),dtype=tf.float32)
    def pack(d,with_teacher=False):
        x={'text':d['text'],'audio':d['audio'],'mask':np.ones((len(d['y']),2),dtype='float32')}
        if with_teacher:x['teacher']=teacher
        return x,d['y'].astype('float32').reshape(-1,1)
    train,validation=pack(tr,True),pack(dev)
    rows=[]
    for mode in ('plain','standard_kd','ra_kd'):
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
        network=architecture(tf,tr['text'].shape[1]);model=Distilled(network,mode)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),run_eagerly=False)
        folder=output/mode;folder.mkdir(parents=True,exist_ok=True)
        stopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=18,min_delta=1e-4,
                                                   restore_best_weights=True,mode='min')
        history=model.fit(*train,validation_data=validation,epochs=args.epochs,batch_size=args.batch_size,
                          verbose=2,shuffle=True,callbacks=[stopping])
        pd.DataFrame(history.history).assign(epoch=np.arange(1,len(history.epoch)+1)).to_csv(folder/'history.csv',index=False)
        network.save_weights(folder/'student.weights.h5')
        results={}
        for name,mask in [('clean',(1,1)),('missing_text',(0,1)),('missing_audio',(1,0))]:
            feed={'text':dev['text'],'audio':dev['audio'],
                  'mask':np.tile(mask,(len(dev['y']),1)).astype('float32')}
            p=network.predict(feed,batch_size=32,verbose=0).reshape(-1)
            results[name]=score(dev['y'],p)
            pd.DataFrame({'participant_id':dev['ids'],'label':dev['y'],'probability':p,
                          'prediction':(p>=.5).astype(int)}).to_csv(folder/f'dev_{name}_predictions.csv',index=False)
        (folder/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
        rows.append({'mode':mode,'epochs_ran':len(history.epoch),
                     **{f'dev_{key}':results['clean'][key] for key in ('accuracy','macro_f1','depressed_f1','auroc')},
                     'missing_text_macro_f1':results['missing_text']['macro_f1'],
                     'missing_audio_macro_f1':results['missing_audio']['macro_f1']})
        print(mode,'DEV',results['clean'],'missing text',results['missing_text']['macro_f1'],
              'missing audio',results['missing_audio']['macro_f1'],flush=True)
    pd.DataFrame(rows).to_csv(output/'comparison.csv',index=False)
    preprocessing.update({'architecture':'Keras two-branch temporal student',
        'framework':tf.__version__,'seed':args.seed,'teacher':teacher_info,
        'selection':'EarlyStopping on weighted DEV loss, not DEV F1',
        'augmentation':'TRAIN feature-space Gaussian noise sigma .05; missing-modality masks: text .25, audio .25, both .5',
        'kd':'TRAIN teacher probabilities; 0.3 soft BCE, 0.7 weighted hard BCE; RA uses entropy-confidence and availability',
        'test_opened':False,'limitation':'DEV used repeatedly to develop method and select models; results exploratory'})
    (output/'protocol.json').write_text(json.dumps(preprocessing,indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
