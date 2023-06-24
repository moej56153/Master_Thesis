import os

revolutions = "2210,2212,2214,2215,2217"

folder_name = "2210_2_4_5_7"


local_path = "./main_files/spimodfit_fits"
d_path = f"{local_path}/{folder_name}"

sample_path = f"{local_path}/sample_scripts"

if not os.path.exists(d_path):
    os.mkdir(d_path)
    
adjust_name = f"adjust4threeML_SE_02_{folder_name}.pro"
background_name = f"background_model_SE_02_{folder_name}.pro"
smf_fit_name = f"spimodfit.fit_Crab_SE_02_{folder_name}.par"
spiselectscw_name = f"spiselectscw.cookbook_dataset_02_0020-0600keV_SE_{folder_name}.par"
    
with open(f"{sample_path}/adjust4threeML_SE_02_1234.pro", "r") as sample:
    with open(f"{d_path}/{adjust_name}", "w") as script:
        for i, line in enumerate(sample):
            if i in [0, 2, 4]:
                line = line.replace("1234", folder_name)
                # line = line.replace("xxxx", folder_name)
            script.write(line)
    
with open(f"{sample_path}/background_model_SE_02_1234.pro", "r") as sample:
    with open(f"{d_path}/{background_name}", "w") as script:
        for i, line in enumerate(sample):
            if i in [9, 10, 16]:
                line = line.replace("1234", folder_name)
                # line = line.replace("xxxx", folder_name)
            script.write(line)
            
with open(f"{sample_path}/spimodfit.fit_Crab_SE_02_1234.par", "r") as sample:
    with open(f"{d_path}/{smf_fit_name}", "w") as script:
        for i, line in enumerate(sample):
            if i in [17, 18, 19, 20, 21, 22]:
                line = line.replace("1234", folder_name)
                # line = line.replace("xxxx", folder_name)
            script.write(line)
            
with open(f"{sample_path}/spiselectscw.cookbook_dataset_02_0020-0600keV_SE_1234.par", "r") as sample:
    with open(f"{d_path}/{spiselectscw_name}", "w") as script:
        for i, line in enumerate(sample):
            if i in [15, 16]:
                line = line.replace("1234", revolutions)
            script.write(line)
            
with open(f"{d_path}/commands.txt", "w") as file:
    file.write("SMF Start-up:\n\n")
    file.write("ssh jmoeller@ga05us.mpe.mpg.de\n")
    file.write("cd /home/jmoeller/cookbook/SPI_cookbook/MT/Crab_fits\n")
    file.write("kinit jmoeller@IPP-GARCHING.MPG.DE\n")
    file.write(". init_ga05us.sh\n\n\n")
    
    file.write("SMF Set-up:\n\n")
    file.write("cd /home/jmoeller/cookbook/SPI_cookbook/MT/Crab_fits\n\n\n")
    
    file.write("Transfer Script to SMF:\n\n")
    file.write(f"cp Master_Thesis/main_files/spimodfit_fits/{folder_name}/{{{adjust_name},{background_name},{smf_fit_name},{spiselectscw_name}}} /mnt/c/Users/moell/Desktop/\n")
    # file.write(f"scp C:\\Users\\moell\\Desktop\\{{{adjust_name},{background_name},{smf_fit_name},{spiselectscw_name}}} jmoeller@ga05us.mpe.mpg.de:/home/jmoeller/cookbook/SPI_cookbook/MT/Crab_fits/\n\n\n")
    file.write("cd Desktop\n")
    file.write(f"scp {adjust_name} {background_name} {smf_fit_name} {spiselectscw_name} jmoeller@ga05us.mpe.mpg.de:/home/jmoeller/cookbook/SPI_cookbook/MT/Crab_fits/\n\n\n")
    
    file.write("SMF Fitting:\n\n")
    file.write(f"./submit-spiselectscw_ga05us.sh cookbook_dataset_02_0020-0600keV_SE_{folder_name} &\n")
    file.write(f"idl idl-startup.pro {background_name}\n")
    file.write(f"./submit-spimodfit_v3.2_ga05us.sh fit_Crab_SE_02_{folder_name} clobber &\n")
    file.write(f"less +F fit_Crab_SE_02_{folder_name}/spimodfit.log\n")
    file.write(f"grep \"Corresponding Pearson's chi2 stat / dof\" fit_Crab_SE_02_{folder_name}/spimodfit.log\n")
    file.write(f"cd fit_Crab_SE_02_{folder_name}\n")
    file.write(f"./spimodfit_rmfgen.csh\n")
    file.write("cd ..\n")
    file.write(f"idl idl-startup.pro {adjust_name}\n\n\n")
    
    file.write("Transfer Results Back:\n\n")
    file.write(f"scp jmoeller@ga05us.mpe.mpg.de:/home/jmoeller/cookbook/SPI_cookbook/MT/Crab_fits/fit_Crab_SE_02_{folder_name}/{{spectral_response.rmf.fits,spectra_Crab_Nebula.fits,spectra_A0535+26a.fits}} C:\\Users\\moell\\Desktop\n")
    file.write(f"cp /mnt/c/Users/moell/Desktop/{{spectral_response.rmf.fits,spectra_Crab_Nebula.fits,spectra_A0535+26a.fits}} Master_Thesis/main_files/spimodfit_fits/{folder_name}\n")
    
    
    
    
    
    
    








