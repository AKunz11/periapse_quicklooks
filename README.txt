This README is for the limb aurora ql scripts (and associated tools) that creates the MAVEN/IUVS limb scan quicklooks. 
Code is written by Zac Milby and Kyle Connour. 
Annotations to it by Jay Cessna and Alexander Kunz
pdf_merger, and mlr_app by Alexander Kunz



=======================================================================================================================================================
Limb aurora QL


Files you need:
1. MAVEN IUVS periapse L1b files, namely muv versions
2. MAVEN IUVS periapse L1c files with NO nightglow features (not given in default L1c files)

What you need to do:
1. Install requirements
   pip install -r requirements.txt

2. Update paths in lines 42-46

3. Update orbit range in lines 38-40

4. Run code and wait a few


Notes:
The CNN version has updated method for cosmic ray (CR) detection, colored as:
   Blue: Convoluted neural network has confidence above threshold in line 33
   Red: MLR constant above the set threshold, or the chi-sq is above the set threshold, or the solar term in MLR above threshold (currently at 100)
   Green: Both the blue and the red are true

   In the scan images, yellow boxes represent integrations removed by CR filtering

   These values are outputted next to each detector image as:
   Integration    chi_sq/chi_sq_threshold    const/const_threshold   solar_term     CNN_prob

The "updated" version does not use all of that, just the const and chi_sq terms, and has a blue overlay over removed integrations

=======================================================================================================================================================
Pdf merger

pdf_merger.py is an optional way to combine the many QL files into a single pdf if desired
   Again, update file paths at the top of the file

=======================================================================================================================================================
MLR Spectrum Viewing

Have you ever wanted to view the MLR fit for an orbit and scan? NOW YOU CAN!
   mlr_app creates an interactive plot to view each of the fit components on the spectra!
   Likely will have to update paths to files in the open_l1b and open_l1c functions
   Run like a normal python script, and then open the link it outputs in a browser