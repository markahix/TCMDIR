# TeraChem Ab Initio MD IR Spectrum Calculator

This program takes as an input the logfile (or output file) of a TeraChem QM/MM MD calculation.  As TeraChem automatically calculates the dipole moment at each timestep of the MD calculation, we can capture these values along with the timestep used and calculate the theoretical IR spectrum of the system given.

#### Usage
`tcmdir -i tc.out -o ir_data.csv`

This command takes input (`-i`) and output (`-o`) flags with their respective filenames, and generates a CSV file with the wavenumbers ($cm^{-1}$) and corresponding intensities.  This file may be used with contemporary plotting software ranging from `matplotlib` to `Excel`, depending on the user's preference.

#### Future Goals
- Implementation of this functionality directly into TeraChem as a keyword-flagged option or just as a part of the MD simulation.
- Expansion of scope to include direct plot generation without requiring external software.