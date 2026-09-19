from pathlib import Path
import argparse,json,subprocess

SOURCES={
 'bias_in_daic-woz':('https://github.com/idiap/bias_in_daic-woz.git','main'),
 'DepressionEstimation':('https://github.com/PingCheng-Wei/DepressionEstimation.git','main')}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--root',default='/content/solo_teacher_sources'); a=p.parse_args(argv); root=Path(a.root); root.mkdir(parents=True,exist_ok=True); resolved={}
    for name,(url,ref) in SOURCES.items():
        dst=root/name
        if not (dst/'.git').exists(): subprocess.run(['git','clone','--depth','1','--branch',ref,url,str(dst)],check=True)
        else: subprocess.run(['git','-C',str(dst),'fetch','--depth','1','origin',ref],check=True); subprocess.run(['git','-C',str(dst),'reset','--hard','FETCH_HEAD'],check=True)
        resolved[name]=subprocess.check_output(['git','-C',str(dst),'rev-parse','HEAD'],text=True).strip(); print(f'{name}: {resolved[name]}')
    (root/'sources.json').write_text(json.dumps(resolved,indent=2)+'\n')

if __name__=='__main__': main()

