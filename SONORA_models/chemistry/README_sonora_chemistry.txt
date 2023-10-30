SONORA CHEMISTRY GRIDS README
Spring 2021
Channon Visscher

THE MODELS HERE ARE SPLIT INTO DIFFERENT GRIDS (BASED UPON THE SAME CALCULATIONS)

1060: The 1060-point SONORA grid of 1060 pressure and temperature (60) points 
1080: Based the Sonora/Marley grid, but with high-P, low-T corner (60 temperatures, 18 pressures)
1920: A grid that covers 75-6000 K and 1E-6 to 3E+3 bar over 96 temperatures and 20 pressures
2020: Includes all grid points calculated (1080 grid + the high-T, high-P 1920 grid)
500k: This grid covers 500-6000 K over 1260 points (63 temperatures, 20 pressures)

GRID NOTES
 * the full equilibrium calculation includes hundreds of additional species; the species
   reported here are a selected output of abundances relevant for opacity calculations
 * unless calculated, gas abundances at low temperatures set to 4.5E-38
 * graphite condensation is included in the calculations (indicated where activity=1); 
   the graphite stability field is also indicated in the 2dplots
 * consideration of ion chemistry for all elements above 500 K
 * PH3 is adopted as the stable low-T P-bearing gas (i.e., JANAF P4O6 data)

CURRENT GRIDS (selected values, see below)
FE/H: -1.0,-0.7,-0.5,-0.3,0.0,+0.3,+0.5,+0.7,+1.0,+1.3,+1.7,+2.0
C/O: 0.25x, 0.5x, 1.0x (solar), 1.5x, 2.0x, 2.5x

METALLICITY VARIATIONS AND THE CARBON-TO-OXYGEN RATIO

The *solar* carbon-to-oxygen ratio is calculated from Lodders (2010):
CARBON = 7.19E6 ATOMS
OXYGEN = 1.57E7 ATOMS
This gives a "solar" C/O ratio of 0.458

The C/O ratio adjusted by keeping C + O = constant and adjusting the carbon-to-oxygen
ratio by a factor relative to the solar value (i.e., a factor of "1" means 1x the solar
value, i.e. a C/O ratio of 0.458). 

This approach keeps the heavy-element-to-hydrogen ratio (Z/X) constant for a given [Fe/H]

These abundances are then multiplied by the metallicity factor (10**[Fe/H]) along
with every other element in the model.


FE/H: -1.0 -0.7 -0.5 -0.3  0.0  0.5  1.0  1.3  1.7  2.0
C/O x:
0.25    X    X    X    X    X    X    X    X    X    X     
0.5     X    X    X    X    X    X    X    X    X    X     
1.0     X    X    X    X    S    X    X    X    X    X     
1.5     X    X    X    X    X    X    X    X    X    X     
2.0     X    X    X    X    X    X    X    X    X    X     
2.5     X    X    X    X    X    X    X    X    X    X 


SOLAR METALLICITY DENOTED BY 'S' ABOVE

METALLICITY AND C/O RATIO INDICATED BY FILENAME

sonora_zzzz_fehxxxx_co_yyy.txt

zzzz = number of grid points
IF zzzz=
1060: Marley et al. (2021) grid


xxxx = METALLICITY
IF xxxx = ...
	-100, THEN [FE/H] = -1.00 (0.1X SOLAR)
	-070, THEN [FE/H] = -0.70
	-050, THEN [FE/H] = -0.50	
	-030, THEN [FE/H] = -0.30	
	+000, THEN [FE/H] =  0.00 (SOLAR)
	+030, THEN [FE/H] = +0.30	
	+050, THEN [FE/H] = +0.50
	+070, THEN [FE/H] = +0.70
	+100, THEN [FE/H] = +1.00 (10X SOLAR)
	+130, THEN [FE/H] = +1.30
	+150, THEN [FE/H] = +1.50	
	+170, THEN [FE/H] = +1.70
	+200, THEN [FE/H] = +2.00 (100X SOLAR)

yyy = C/O RATIO
IF yyy = ...
	025, THEN C/O = 0.25X SOLAR
	050, THEN C/O = 0.5X SOLAR
	100, THEN C/O = 1.0X SOLAR (SOLAR C/O = 0.458)_
	150, THEN C/O = 1.5X SOLAR
	200, THEN C/O = 2.0X SOLAR
	250, THEN C/O = 2.5X SOLAR	

