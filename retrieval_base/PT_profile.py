import numpy as np
from scipy.interpolate import interp1d, splrep, splev, RegularGridInterpolator#, LinearNDInterpolator
from scipy.interpolate import make_interp_spline

import petitRADTRANS.poor_mans_nonequ_chem as pm

class PT_profile:

    def __init__(self, pressure):
        
        self.pressure = pressure
        self.ln_L_penalty = 0

class PT_profile_SONORA(PT_profile):

    def __init__(self, pressure, path_to_SONORA_PT='./data/SONORA_PT_structures'):

        # Give arguments to the parent class
        super().__init__(pressure)

        import glob, os
        all_paths = glob.glob(os.path.join(path_to_SONORA_PT, '*/t*.dat'))
        all_paths = np.sort(all_paths)

        # T_eff, log_g, C/O, Fe/H
        self.T_eff_grid = np.nan * np.ones(all_paths.shape)
        self.log_g_grid = np.nan * np.ones(all_paths.shape)
        self.CO_grid = np.nan * np.ones(all_paths.shape)
        self.FeH_grid = np.nan * np.ones(all_paths.shape)

        for i, path_i in enumerate(all_paths):
            # Read from the file headers
            T_eff_i, g_i, FeH_i, CO_i = \
                np.loadtxt(path_i, max_rows=1, usecols=(0,1,5,6))
            T_eff_i = np.around(T_eff_i, -1)

            # Convert to cgs and take log10
            log_g_i = np.log10(g_i*1e2)
            log_g_i = np.around(log_g_i, 2)
            
            # Convert from Solar to actual C/O
            CO_i = CO_i*0.6

            self.T_eff_grid[i] = T_eff_i
            self.log_g_grid[i] = log_g_i
            self.CO_grid[i]    = CO_i
            self.FeH_grid[i]   = FeH_i

        # Get only one of each entry
        self.T_eff_grid = np.sort(np.unique(self.T_eff_grid))
        self.log_g_grid = np.sort(np.unique(self.log_g_grid))
        self.CO_grid    = np.sort(np.unique(self.CO_grid))
        self.FeH_grid   = np.sort(np.unique(self.FeH_grid))

        self.temperature_grid = np.nan * np.ones(
            (len(self.T_eff_grid), len(self.log_g_grid), 
             len(self.CO_grid), len(self.FeH_grid), 50)
            )

        self.rad_conv_boundary_grid = np.nan * np.ones(
            (len(self.T_eff_grid), len(self.log_g_grid), 
             len(self.CO_grid), len(self.FeH_grid))
            )
        
        for i, path_i in enumerate(all_paths):
            # Read from the file headers
            T_eff_i, g_i, FeH_i, CO_i = \
                np.loadtxt(path_i, max_rows=1, usecols=(0,1,5,6))
            T_eff_i = np.around(T_eff_i, -1)

            # Convert to cgs and take log10
            log_g_i = np.log10(g_i*1e2)
            log_g_i = np.around(log_g_i, 2)
            
            # Convert from Solar to actual C/O
            CO_i = CO_i*0.6

            mask_T_eff = (self.T_eff_grid == T_eff_i)
            mask_log_g = (self.log_g_grid == log_g_i)
            mask_CO    = (self.CO_grid == CO_i)
            mask_FeH   = (self.FeH_grid == FeH_i)

            # Load the PT profile
            pressure_i, temperature_i, ad_gradient_i, rad_gradient_i = np.genfromtxt(
                path_i, skip_header=1, usecols=(1,2,4,5), delimiter=(3,12,10,11,8,8)
                ).T

            # Schwarzschild criterion
            self.rad_conv_boundary_grid[mask_T_eff,mask_log_g,mask_CO,mask_FeH] = \
                pressure_i[(ad_gradient_i < rad_gradient_i)].min()

            self.temperature_grid[mask_T_eff,mask_log_g,mask_CO,mask_FeH,:] = interp1d(
                pressure_i, temperature_i, kind='cubic', fill_value='extrapolate'
                )(self.pressure)

        # Set-up an function to conduct 4D-interpolation
        points = ()
        if len(self.T_eff_grid) > 1:
            points += (self.T_eff_grid, )
        if len(self.log_g_grid) > 1:
            points += (self.log_g_grid, )
        if len(self.CO_grid) > 1:
            points += (self.CO_grid, )
        if len(self.FeH_grid) > 1:
            points += (self.FeH_grid, )

        self.temperature_grid = np.squeeze(self.temperature_grid)
        self.rad_conv_boundary_grid = np.squeeze(self.rad_conv_boundary_grid)

        self.temperature_interp_func = RegularGridInterpolator(
            points=points, values=self.temperature_grid
            )
        
        self.rad_conv_boundary_interp_func = RegularGridInterpolator(
            points=points, values=self.rad_conv_boundary_grid
            )
        
    def __call__(self, params):

        # Interpolate the 4D grid onto the requested point
        point = ()
        if len(self.T_eff_grid) > 1:
            point += (params['T_eff'], )
        if len(self.log_g_grid) > 1:
            point += (params['log_g'], )
        if len(self.CO_grid) > 1:
            point += (params['C/O'], )
        if len(self.FeH_grid) > 1:
            point += (params['Fe/H'], )

        self.temperature = self.temperature_interp_func(point)
        self.RCB = self.rad_conv_boundary_interp_func(point)

        return self.temperature
    
class PT_profile_free(PT_profile):

    def __init__(self, pressure, ln_L_penalty_order=3, PT_interp_mode='log'):
        
        # Give arguments to the parent class
        super().__init__(pressure)

        self.ln_L_penalty_order = ln_L_penalty_order
        self.PT_interp_mode = PT_interp_mode

    def __call__(self, params):

        # Combine all temperature knots
        self.T_knots = np.concatenate((params['T_knots'], 
                                       [params['T_0']]
                                       ))

        self.P_knots = params['P_knots']

        if self.P_knots is None:
            # Use evenly-spaced (in log) pressure knots
            self.P_knots = np.logspace(np.log10(self.pressure.min()), 
                                       np.log10(self.pressure.max()), 
                                       num=len(self.T_knots))

        # Log-likelihood penalty scaling factor
        self.gamma = params['gamma']

        # Cubic spline interpolation over all layers
        self.spline_interp()
        # Compute the log-likelihood penalty
        if self.gamma is not None:
            self.get_ln_L_penalty()
            
        return self.temperature

    def spline_interp(self):

        if self.PT_interp_mode == 'log':
            y = np.log10(self.T_knots)
        elif self.PT_interp_mode == 'lin':
            y = self.T_knots

        # Spline interpolation over a number of knots
        #spl = make_interp_spline(np.log10(self.P_knots), y, bc_type=([(3,0)],[(3,0)]))
        spl = make_interp_spline(np.log10(self.P_knots), y, bc_type='not-a-knot')

        if self.PT_interp_mode == 'log':
            self.temperature = 10**spl(np.log10(self.pressure))
        elif self.PT_interp_mode == 'lin':
            self.temperature = spl(np.log10(self.pressure))

        self.coeffs = spl.c
        self.knots  = spl.t

        # Remove padding zeros
        self.coeffs = self.coeffs[:len(self.knots)-4]

    def get_ln_L_penalty(self):

        # Compute the log-likelihood penalty based on the wiggliness
        # (Inverted) weight matrices, scaling the penalty of small/large segments
        inv_W_1 = np.diag(1/(1/3 * np.array([self.knots[i+3]-self.knots[i] \
                                             for i in range(1, len(self.knots)-4)]))
                          )
        inv_W_2 = np.diag(1/(1/2 * np.array([self.knots[i+2]-self.knots[i] \
                                             for i in range(2, len(self.knots)-4)]))
                          )
        inv_W_3 = np.diag(1/(1/1 * np.array([self.knots[i+1]-self.knots[i] \
                                             for i in range(3, len(self.knots)-4)]))
                          )

        # Fundamental difference matrix
        delta = np.zeros((len(inv_W_1), len(inv_W_1)+1))
        delta[:,:-1] += np.diag([-1]*len(inv_W_1))
        delta[:,1:]  += np.diag([+1]*len(inv_W_1))

        # 1st, 2nd, 3rd order general difference matrices
        D_1 = np.dot(inv_W_1, delta)
        D_2 = np.dot(inv_W_2, np.dot(delta[1:,1:], D_1))
        D_3 = np.dot(inv_W_3, np.dot(delta[2:,2:], D_2))
        
        # General difference penalty, computed with L2-norm
        if self.ln_L_penalty_order == 1:
            gen_diff_penalty = np.nansum(np.dot(D_1, self.coeffs)**2)
        elif self.ln_L_penalty_order == 2:
            gen_diff_penalty = np.nansum(np.dot(D_2, self.coeffs)**2)
        elif self.ln_L_penalty_order == 3:
            gen_diff_penalty = np.nansum(np.dot(D_3, self.coeffs)**2)

        self.ln_L_penalty = -(1/2*gen_diff_penalty/self.gamma + \
                              1/2*np.log(2*np.pi*self.gamma)
                              )

class PT_profile_Kitzmann(PT_profile):

    def __init__(self, pressure, order=2):

        # Give arguments to the parent class
        super().__init__(pressure)

        # Set the piecewise polynomial orders
        self.order = order
        # Number of local grid points
        self.N_p = order + 1

    def __call__(self, params):

        # Combine all temperature knots
        self.T_knots = np.concatenate((params['T_knots'], 
                                       [params['T_0']]
                                       ))

        self.P_knots = params['P_knots']

        if self.P_knots is None:
            # Use evenly-spaced (in log) pressure knots
            self.P_knots = np.logspace(np.log10(self.pressure.min()), 
                                       np.log10(self.pressure.max()), 
                                       num=len(self.T_knots))

        self.log_P_knots = np.log10(self.P_knots)

        # De-construct into piecewise elements
        self.log_P_knots_local = []
        self.T_knots_local = []
        
        for k in range(0, len(self.T_knots)-self.N_p+1, self.N_p-1):
            # Add knots based on the polynomial order
            self.T_knots_local.append(self.T_knots[k:k+self.N_p])
            self.log_P_knots_local.append(self.log_P_knots[k:k+self.N_p])
            
        # Perform a piecewise polynomial interpolation
        self.piecewise_interp()

    def piecewise_interp(self):

        def _basis(i, k):
            # Lagrange polynomials
            l_i = np.prod([
                (np.log10(self.pressure) - self.log_P_knots_local[k][j]) / \
                    (self.log_P_knots_local[k][i] - self.log_P_knots_local[k][j])
                for j in range(0, self.N_p) if j != i
                ], axis=0)
            
            return l_i

        def _sum(k):
            # Sum of Lagrange polynomials
            T_k = np.sum([
                _basis(i, k) * self.T_knots_local[k][i]
                for i in range(0, self.N_p)
                ], axis=0)
            
            return T_k

        self.temperature = np.ones_like(self.pressure) * np.nan
        for k in range(len(self.T_knots_local)):

            # Only add to specific pressure range
            if k == 0:
                # Including both knots
                element_mask = (np.log10(self.pressure) >= self.log_P_knots_local[k].min()) & \
                    (np.log10(self.pressure) <= self.log_P_knots_local[k].max())
            else:
                element_mask = (np.log10(self.pressure) > self.log_P_knots_local[k].min()) & \
                    (np.log10(self.pressure) <= self.log_P_knots_local[k].max())

            self.temperature[element_mask] = _sum(k)[element_mask]
        
        return self.temperature
        
class PT_profile_Molliere(PT_profile):

    def __init__(self, pressure, conv_adiabat=True):

        # Give arguments to the parent class
        super().__init__(pressure)

        # Go from bar to cgs
        self.pressure_cgs = self.pressure * 1e6

        self.conv_adiabat = conv_adiabat

    def pressure_tau(self, tau):
        
        # Return the pressure at a given optical depth, in cgs
        return tau**(1/self.alpha) * self.P_phot

    def photosphere(self):
        # Middle altitudes

        # Calculate the optical depth
        self.tau = (self.pressure_cgs/self.P_phot)**self.alpha

        # Eddington temperature at middle altitudes
        self.T_photo = (3/4 * self.T_int**4 * (2/3 + self.tau))**(1/4)

    def troposphere(self):
        # Low altitudes

        if self.conv_adiabat and ((self.CO is None) or (self.FeH is None)):
            # conv_adiabat requires equilibrium chemistry to compute 
            # adiabatic gradient, use Eddington approximation instead
            self.conv_adiabat = False

        if self.conv_adiabat:
            # Enforce convective adiabat at low altitudes

            # Retrieve the adiabatic temperature gradient
            nabla_ad = pm.interpol_abundances(CO*np.ones_like(self.T_photo), 
                                              FeH*np.ones_like(self.T_photo), 
                                              self.T_photo, self.pressure
                                              )['nabla_ad']

            # Calculate the current radiative temperature gradient
            nabla_rad_temporary = np.diff(np.log(self.T_photo)) / np.diff(np.log(self.pressure_cgs))

            # Extend the array to the same length as pressure structure
            nabla_rad = np.ones_like(self.T_photo)
            nabla_rad[[0,-1]] = nabla_rad_temporary[[0,-1]] # Edges are the same
            nabla_rad[1:-1] = (nabla_rad_temporary[1:] + nabla_rad_temporary[:-1]) / 2 # Interpolate

            # Mask where atmosphere is convectively (Schwarzschild)-unstable
            mask_unstable = (nabla_rad > nabla_ad)

            for i in range(10):

                if i == 0:
                    # Initially, use the eddington approximation
                    T_to_use = self.T_photo.copy()
                else:
                    T_to_use = self.T_tropo.copy()

                # Retrieve the adiabatic temperature gradient
                nabla_ad = pm.interpol_abundances(CO*np.ones_like(T_to_use), 
                                                  FeH*np.ones_like(T_to_use), 
                                                  T_to_use, self.pressure
                                                  )['nabla_ad']
                
                # Calculate the average adiabatic temperature gradient between the layers
                nabla_ad_mean = nabla_ad
                nabla_ad_mean[1:] = (nabla_ad[1:] + nabla_ad[-1]) / 2

                # What are the increments in temperature due to convection
                log_T_new = nabla_ad_mean[mask_unstable] * np.mean(np.diff(np.log(self.pressure_cgs)))

                # What is the last radiative temperature
                log_T_start = np.log(T_to_use[~mask_unstable][-1])

                # Integrate and translate to temperature from log(T)
                T_new = np.exp(np.cumsum(log_T_new) + log_T_start)

                # Combine upper radiative and lower convective part into an array
                self.T_tropo = T_to_use.copy()
                self.T_tropo[mask_unstable] = T_new

                if np.max(np.abs(T_to_use - self.T_tropo) / T_to_use) < 0.01:
                    break
        
        else:
            # Otherwise, use Eddington approximation at low altitudes
            self.T_tropo = self.T_photo

    def high_altitudes(self):
        # High altitudes, add the 3 point PT description above tau=0.1
        
        # Uppermost pressure of the Eddington radiative structure
        P_bottom_spline = self.pressure_tau(0.1)

        # Apply two iterations of spline interpolation to smooth out transition
        for i in range(2):

            if i == 0:
                num = 4
            else:
                num = 7
                    
            # Create the pressure coordinates for the spline support nodes at low pressure
            P_support_low = np.logspace(np.log10(self.pressure_cgs[0]),
                                        np.log10(P_bottom_spline),
                                        num=num
                                        )

            # Create the pressure coordinates for the spline support nodes at high pressure,
            # the corresponding temperatures for these nodes will be taken from the 
            # radiative + convective solution
            if i == 0:
                P_support_high = 10**np.arange(np.log10(P_bottom_spline), 
                                               np.log10(self.pressure_cgs[-1]),
                                               np.diff(np.log10(P_support_low))[0]
                                               )
            else:
                # Use equal number of support points at low altitude
                P_support_high = np.logspace(np.log10(P_bottom_spline), 
                                             np.log10(self.pressure_cgs[-1]),
                                             num=min([num, 7])
                                             )

            # Combine into one support node array, only adding the P_bottom_spline point once
            P_support = np.concatenate((P_support_low, P_support_high[1:]))

            # Define the temperature values at the node points
            T_support = np.zeros_like(P_support)

            if i == 0:
                # Define an interpolation function
                interp_func_T_tropo = interp1d(self.pressure_cgs, self.tropo)

                # Temperature at pressures below P_bottom_spline (free parameters)
                T_support[:len(self.T_knots_init)] = self.T_knots_init

            else:
                # Define an interpolation function
                interp_func_temperature = interp1d(self.pressure_cgs, self.temperature)

                # Temperature at pressures below P_bottom_spline
                T_support[:len(P_support_low)-1] = interp_func_temperature(P_support[:len(P_support_low)-1])

            # Temperature at pressures at or above P_bottom_spline (from the radiative-convective solution)
            T_support[-len(P_support_high):] = interp_func_T_tropo(P_support[-len(P_support_high):])

            # Make the temperature spline interpolation
            knots, coeffs, deg = splrep(np.log10(P_support), T_support)
            self.temperature = splev(np.log10(self.pressure_cgs), (knots, coeffs, deg), der=0)
    
            if i == 0:
                # Go from cgs to bar
                self.P_knots = P_support/1e6
                self.T_knots = T_support

    def __call__(self, params):

        # Update the parameters
        # Convert from bar to cgs
        self.P_phot = params['P_phot'] * 1e6

        self.T_knots_init = params['T_knots']
        self.T_int = params['T_int']
        self.alpha = params['alpha']

        self.CO  = params['C/O']
        self.FeH = params['Fe/H']

        # Calculate for each segment of the atmosphere
        self.photosphere()
        self.troposphere()
        self.high_altitudes()

        return self.temperature
