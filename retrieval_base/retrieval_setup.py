from pathlib import Path

from . import utils

class RetrievalSetup:

    def __init__(self, config):

        # Create output directory
        self.prefix = config.prefix
        self.create_output_dir(config.file_params)
        
        # Pre-process the data or generate synthetic spectrum
        if hasattr(config, 'config_data'):
            
            self.model_settings = list(config.config_data.keys())

            for m_set_i, config_data_i in config.config_data.items():
                # Loop over model settings
                self.pre_process_data(config_data_i)

        elif hasattr(config, 'config_synthetic_spectrum'):

            self.model_settings = list(config.config_synthetic_spectrum.keys())

            for m_set_i, config_synthetic_spectrum_i in config.config_synthetic_spectrum.items():
                # Loop over model settings
                self.get_synthetic_spectrum(config_synthetic_spectrum_i, m_set_i)

        else:
            raise ValueError('config must have either config_data or config_synthetic_spectrum')

        # Get a parameters object
        self.get_parameters(config)

        # TODO: Generate pRT_model object
        # Save
        pass

    def create_output_dir(self, file_params):

        # Create output directory
        self.data_dir = Path(f'{self.prefix}data')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = Path(f'{self.prefix}plots')
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy config-file to output directory
        config_file = Path(file_params)
        destination = self.data_dir / config_file.name
        destination.write_bytes(config_file.read_bytes())
        
    def pre_process_data(self, config_data):

        from .data.spectrum_crires import DataSpectrumCRIRES

        # Load the target and standard-star spectra
        d_spec_target = DataSpectrumCRIRES(
            **config_data['target_kwargs'], **config_data['kwargs']
            )
        d_spec_std = DataSpectrumCRIRES(
            **config_data['std_kwargs'], **config_data['kwargs']
            )
        
        # Pre-process the data
        d_spec_target.telluric_correction(d_spec_std)
        d_spec_target.flux_calibration(**config_data['target_kwargs'])
        d_spec_target.sigma_clip(**config_data['kwargs'])
        d_spec_target.savgol_filter()

        # Summarise in figures and save
        d_spec_target.make_figures(plots_dir=self.plots_dir)
        d_spec_target.save_to_pickle(data_dir=self.data_dir)
        
        self.d_spec_target = d_spec_target

    def get_synthetic_spectrum(self, config_synthetic_spectrum, m_set):
        raise NotImplementedError
    
    def get_parameters(self, config):

        from .parameters_new import ParameterTable

        # Create a Parameter instance
        self.ParamTable = ParameterTable(
            free_params=config.free_params, 
            constant_params=config.constant_params, 
            model_settings=self.model_settings, 

            PT_kwargs=config.PT_kwargs, 
            chem_kwargs=config.chem_kwargs, 
            cloud_kwargs=config.cloud_kwargs, 
            cov_kwargs=config.cov_kwargs, 
            )

        # Save the Parameter instance
        utils.save_pickle(self.ParamTable, self.data_dir / 'ParamTable.pkl')

    def get_model(self):

        # Create the model object

        pass