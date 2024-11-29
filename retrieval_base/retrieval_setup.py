from pathlib import Path

class RetrievalSetup:

    def __init__(self, config):

        # Create output directory
        self.prefix = config.prefix
        self.create_output_dir(config.file_params)
        
        # Pre-process the data or generate synthetic spectrum
        if hasattr(config, 'config_data'):
            for m_set_i, config_data_i in config.config_data.items():
                # Loop over model settings
                self.pre_process_data(config_data_i, m_set_i)

        elif hasattr(config, 'config_synthetic_spectrum'):
            for m_set_i, config_synthetic_spectrum_i in config.config_synthetic_spectrum.items():
                # Loop over model settings
                self.get_synthetic_spectrum(config_synthetic_spectrum_i, m_set_i)
                
        else:
            raise ValueError('config must have either config_data or config_synthetic_spectrum')

        # TODO: Generate pRT_model object
        # Save
        pass

    def create_output_dir(self, file_params):

        # Create output directory
        self.data_dir  = Path(f'{self.prefix}data')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = Path(f'{self.prefix}plots')
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy config-file to output directory
        config_file = Path(file_params)
        destination = self.data_dir / config_file.name
        destination.write_bytes(config_file.read_bytes())
        
    def pre_process_data(self, config_data, m_set):

        from .spectrum_crires import DataSpectrumCRIRES
        d_spec_target = DataSpectrumCRIRES(
            **config_data['target_kwargs'], **config_data['kwargs']
            )
        d_spec_std = DataSpectrumCRIRES(
            **config_data['std_kwargs'], **config_data['kwargs']
            )
        
        # Correct for telluric absorption
        d_spec_target.telluric_correction(d_spec_std)

        # Flux calibration
        d_spec_target.flux_calibration(**config_data['target_kwargs'])

        # Sigma-clipping
        d_spec_target.sigma_clip(**config_data['kwargs'])

        # High-pass filtering
        d_spec_target.savgol_filter()

        # Make figures
        d_spec_target.make_figures(plots_dir=self.plots_dir)

        # Save
        # TODO: ...

    def get_synthetic_spectrum(self, config_synthetic_spectrum, m_set):
        raise NotImplementedError

    def get_pRT_model(self):
        pass