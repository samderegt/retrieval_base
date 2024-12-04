from pathlib import Path

from . import utils

class Retrieval:
    def __init__(self, config):
        # Create output directory
        self.create_output_dir(config.prefix, config.file_params)
        self.model_settings = list(config.config_data.keys())

    def create_output_dir(self, prefix, file_params):
        """
        Create output directories for data and plots, and copy the config file.

        Args:
            prefix (str): Prefix for the directory names.
            file_params (str): Path to the configuration file.
        """
        # Create output directory
        self.data_dir = Path(f'{prefix}data')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = Path(f'{prefix}plots')
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy config-file to output directory
        config_file = Path(file_params)
        destination = self.data_dir / config_file.name
        destination.write_bytes(config_file.read_bytes())

    def save_all_components(self):

        for name, component in vars(self).items():
            if not isinstance(component, dict):
                # Skip non-dictionary attributes
                continue

            for m_set, comp in component.items():
                utils.save_pickle(comp, self.data_dir/f'{name}_{m_set}.pkl')

    def load_components(self, component_names):
        
        for name in component_names:
            # Load the component, if it exists
            component = {}
            for m_set in self.model_settings:
                component[m_set] = utils.load_pickle(
                    self.data_dir/f'{name}_{m_set}.pkl'
                    )
            # Make attribute
            setattr(self, name, component)

class RetrievalSetup(Retrieval):
    """
    A class to set up the retrieval.
    """

    def __init__(self, config):
        """
        Initialize the RetrievalSetup instance.

        Args:
            config (object): Configuration object containing necessary parameters.
        """
        # Give arguments to the parent class
        super().__init__(config)
        
        # Pre-process the data or generate synthetic spectrum
        if hasattr(config, 'config_data'):
            self.get_data_spectrum(config.config_data)
        elif hasattr(config, 'config_synthetic_spectrum'):
            raise NotImplementedError
            self.get_synthetic_spectrum(config.config_synthetic_spectrum)
        else:
            raise ValueError('config must have either config_data or config_synthetic_spectrum')

        # Get parameters and model components
        self.get_parameters(config)
        self.get_model_components()
        self.save_all_components()

    def get_data_spectrum(self, config_data):
        """
        Pre-process the data spectrum.

        Args:
            config_data (dict): Configuration data for the spectrum.
        """
        from .data.crires import SpectrumCRIRES

        self.d_spec = {}

        for m_set, config_data_m_set in config_data.items():

            # Load the target and standard-star spectra
            d_spec_target = SpectrumCRIRES(
                **config_data_m_set['target_kwargs'], 
                **config_data_m_set['kwargs']
                )
            d_spec_std = SpectrumCRIRES(
                **config_data_m_set['std_kwargs'], 
                **config_data_m_set['kwargs']
                )
            
            # Pre-process the data
            d_spec_target.telluric_correction(d_spec_std)
            d_spec_target.flux_calibration(**config_data_m_set['target_kwargs'])
            d_spec_target.sigma_clip(**config_data_m_set['kwargs'])
            d_spec_target.savgol_filter()

            # Summarise in figures and store in dictionary
            d_spec_target.make_figures(plots_dir=self.plots_dir)
            self.d_spec[m_set] = d_spec_target

    def get_synthetic_spectrum(self, config_synthetic_spectrum, m_set):
        """
        Generate synthetic spectrum. Not implemented.

        Args:
            config_synthetic_spectrum (dict): Configuration for synthetic spectrum.
            m_set (str): Model setting identifier.
        """
        raise NotImplementedError
    
    def get_parameters(self, config):
        """
        Get parameters for the retrieval process.

        Args:
            config (object): Configuration object containing necessary parameters.
        """
        from .parameters import ParameterTable

        # Create a Parameter instance
        self.ParamTable = ParameterTable(
            free_params=config.free_params, 
            constant_params=config.constant_params, 
            model_settings=self.model_settings, 
            all_model_kwargs=config.all_model_kwargs,
            )

        # Save the Parameter instance
        utils.save_pickle(self.ParamTable, self.data_dir/'ParamTable.pkl')

    def get_model_components(self, evaluation=False):
        """
        Generate model components for the retrieval process.

        Args:
            evaluation (bool): Flag to indicate if it's for evaluation. Default is False.
        """
        self.LineOpacity = {}
        self.PT = {}
        self.Chem = {}
        self.Cloud = {}
        self.Rotation = {}
        self.m_spec = {}
        self.LogLike = {}
        self.Cov = {}
        
        for m_set in self.model_settings:

            # Query the right model settings
            self.ParamTable.queried_m_set = ['all', m_set]

            # Set the physical model components
            from .model_components import pt_profile
            self.PT[m_set] = pt_profile.get_class(
                **self.ParamTable.PT_kwargs[m_set]
                )
            pressure = self.PT[m_set].pressure

            from .model_components import line_opacity
            self.LineOpacity[m_set] = line_opacity.get_class()

            from .model_components import chemistry
            self.Chem[m_set] = chemistry.get_class(
                pressure=pressure, 
                LineOpacity=self.LineOpacity[m_set], 
                **self.ParamTable.chem_kwargs[m_set]
                )
            
            from .model_components import clouds
            self.Cloud[m_set] = clouds.get_class(
                pressure=pressure, 
                Chem=self.Chem[m_set], 
                PT=self.PT[m_set], 
                **self.ParamTable.cloud_kwargs[m_set]
                )

            from .model_components import rotation_profile
            self.Rotation[m_set] = rotation_profile.get_class(
                **self.ParamTable.rotation_kwargs[m_set]
                )

            from .model_components import model_spectrum
            self.m_spec[m_set] = model_spectrum.get_class(
                ParamTable=self.ParamTable, 
                d_spec=self.d_spec[m_set], 
                m_set=m_set,
                pressure=pressure, 
                evaluation=evaluation
                )
            
            # Set the observation model components
            from .model_components import log_likelihood
            self.LogLike[m_set] = log_likelihood.get_class(
                d_spec=self.d_spec[m_set], 
                **self.ParamTable.loglike_kwargs[m_set]
                )
            
            from .model_components import covariance
            self.Cov[m_set] = covariance.get_class(
                d_spec=self.d_spec[m_set], 
                **self.ParamTable.cov_kwargs[m_set]
                )

        self.ParamTable.queried_m_set = 'all'

class RetrievalRun(Retrieval):
    """
    A class to run the retrieval process, including loading data, parameters, and models,
    and initiating the retrieval.
    """

    def __init__(self, config, evaluation=False):
        
        """
        Initialize the RetrievalRun instance.
        """
        # Give arguments to the parent class
        super().__init__(config)

        # Load a list of components
        component_names = [
            'd_spec', 'ParamTable', 
            'LineOpacity', 'PT', 'Chem', 'Cloud', 
            'Rotation', 'm_spec', 'LogLike', 'Cov'
        ]
        '''
        if evaluation:
            component_names += ['m_spec_eval']
        else:
            component_names += ['m_spec']
        '''
        self.load_components(component_names)
    
        # Initiate the retrieval
        self.run()

    def run(self):
        """
        Run the retrieval process.
        """
        # TODO: call pymultinest
        pass

    def evaluation(self):
        """
        Evaluate the results of the retrieval process.
        """
        # TODO: Evaluate the results
        pass