"""Focused numerical checks for the train-only fusion gate."""
import unittest
import numpy as np
from scipy.optimize import check_grad
from scripts.experiments.internal188.expert_gate import objective,fit,fuse


class GateTests(unittest.TestCase):
    def setUp(self):
        r=np.random.default_rng(103);n=107
        self.p=r.uniform(.1,.9,(n,2));self.t=r.uniform(.1,.9,(n,2))
        self.y=np.array([0]*77+[1]*30)
        self.x=np.column_stack([np.ones(n),r.normal(size=(n,2))])

    def test_gradient(self):
        for alpha in (0.,.3):
            args=(self.x,self.p,self.y,self.t.mean(1),alpha)
            error=check_grad(lambda z:objective(z,*args)[0],lambda z:objective(z,*args)[1],np.array([.2,-.1,.05]))
            self.assertLess(error,1e-6)

    def test_fitted_gate_preserves_available_expert(self):
        for mode in ('plain','standard_kd','ra_kd'):
            theta,status=fit(self.x,self.p,self.y,self.t,mode)
            self.assertTrue(status['optimizer_success'])
            for mask,expected in [([1,0],self.p[:,0]),([0,1],self.p[:,1]),([0,0],np.full(107,.28))]:
                np.testing.assert_allclose(fuse(theta,self.x,self.p,np.tile(mask,(107,1)),.28),expected,atol=1e-12)

    def test_fusion_stays_between_expert_probabilities(self):
        q=fuse(np.array([.3,-1.,2.]),self.x,self.p)
        self.assertTrue(np.all(q>=self.p.min(1)-1e-12))
        self.assertTrue(np.all(q<=self.p.max(1)+1e-12))

if __name__=='__main__':unittest.main()
