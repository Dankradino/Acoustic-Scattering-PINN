   _____  ____  _   _ _____ _____ ____  __  __ 
  / ____|/ __ \| \ | |_   _/ ____/ __ \|  \/  |
 | (___ | |  | |  \| | | || |   | |  | | \  / |
  \___ \| |  | | . ` | | || |   | |  | | |\/| |
  ____) | |__| | |\  |_| || |___| |__| | |  | |
 |_____/ \____/|_| \_|_____\_____\____/|_|  |_|
                                               
This folder contains individual Head Related Transfer Function (HRTF) measurements and 3D head scans from the SONICOM HRTF database.

Folder structure:

 - '3DSCAN' contains a 3D scan of the head and torso:

        - 'PXXXX.stl' is the result of the scan as a 3D mesh. You can import this file into nearly all 3D modelling software.
	  Some minor automated patching has been performed, but if you want to use it, for instance for 3D printing, 
          additional post-processing to fill in the holes is advised.
        - 'PXXXX_Project1.asc' contains a point cloud, an intermediate storage format consisting of individual 3D points before they are joined into a mesh.

 - 'HRTF' contains the audio measurements including raw audio, settings, log files and a helper MATLAB script to regenerate the HRTF files:

        - 'HRTF' subfolder contains processed HRTF files at several sample rates (44.1 kHz, 48 kHz, and 96 kHz, divided into individual folders).
          The files are stored in SOFA (https://www.sofaconventions.org) and 3dti formats. Descriptions are included below.

                * FOR BEGINNERS: we recommend trying 'PXXXX_FreeFieldCompMinPhase_NoITD_44kHz.3dti-hrtf' in the 3DTI Binaural Test Application 
                  (https://github.com/3DTune-In/3dti_AudioToolkit/releases/latest).

                - 'PXXXX_Raw_XXkHz': raw measured HRTF (50 ms long, no fade in/out applied);
                - 'PXXXX_Windowed_XXkHz': same as Raw, but windowed to 5 ms and with fade in/out applied;
                - 'PXXXX_FreeFieldComp_XXkHz': same as Windowed, but free-field compensation is applied via a linear-phase EQ filter;
                - 'PXXXX_FreeFieldCompMinPhase_XXkHz': same as Windowed, but free-field compensation is applied via a minimum-phase EQ filter.

                For each file, there is also a 'NoITD' version which has the ITDs removed and is compatible with the 3DTI Toolkit.
                This was done by time-shifting each measurement so that all the onsets were aligned. 
                The amount of shifted samples is stored as metadata in the SOFA file.

        - 'HPEQ' subfolder contains personal equalisation for Sennheiser HD 650 headphones in 44.1 kHz, 48 kHz, and 96 kHz sample rates.
                - 'PXXXX_headphoneEQ_XXkHz.mat': contains data of 5 headphone measurements and minimum-phase EQ filters for both ears;
                - 'PXXXX_headphoneEQ_XXkHz.wav': audio file with the impulse response of a single-channel minimum-phase EQ filter 
                                                 (average of left and right magnitude responses).

 

------------------------------------------
Audio Experience Design, Dyson School of Design Engineering, Imperial College London
https://www.axdesign.co.uk

README v0.2, 01/08/2024
