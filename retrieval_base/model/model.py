
class Model:

    def __init__(self, ParamTable, d_spec, model_settings, evaluation=False):
        
        # Likelihood and data parameters
        self.LogLike = {}
        self.Cov     = {}
        
        # Physical parameters
        self.PT      = {}
        self.Chem    = {}
        self.Cloud   = {}

        # Radiative transfer + spectrum
        self.pRT = {}
        self.Rot = {}
        self.LineOpacity     = {}
        self.InstrBroadening = {}

        from .pRT import pRT
        #self.pRT = pRT(ParamTable, d_spec, m_set, pressure, callback=False)
        pass

    def forward(self, x):
        pass
        #return model spectrum

    #def log_likelihood(self, x):
    #    pass
    #    #return log-likelihood

    #def __call__(self, *args, **kwds):
    #    pass
