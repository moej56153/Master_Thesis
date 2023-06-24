; #######################################
; #######################################
; ####### BACKGROUND MODEL SCRIPT #######
; #######################################
; #######################################

; #--------------------------------------
; # INPUT DATA SET
; #--------------------------------------
spidir = '/home/jmoeller/cookbook/SPI_cookbook/MT/Crab_fits/cookbook_dataset_02_0020-0600keV_SE_1593_7_8_9/spi/'
scw_file = '/home/jmoeller/cookbook/SPI_cookbook/MT/Crab_fits/cookbook_dataset_02_0020-0600keV_SE_1593_7_8_9/scw.fits.gz'
; #--------------------------------------

; #--------------------------------------
; # BACKGROUND DIRECTORY (WILL BE CREATED)
; #--------------------------------------
bgdir = '/home/jmoeller/cookbook/SPI_cookbook/MT/Crab_fits/cookbook_dataset_02_0020-0600keV_SE_1593_7_8_9/spi/bg'
; #--------------------------------------

; #--------------------------------------
; # ENERGY BINS (MUST BE AS IN DATA SET)
; #--------------------------------------
emin = 20.
emax = 600.

; #--------------------------------------
; # TRACER (USE GESATTOT IF NO IDEA)
; #--------------------------------------
tracer = 'GESATTOT'



; ##################################################################
; # DON'T TOUCH ANYTHING BELOW UNLESS YOU KNOW WHAT YOU ARE DOING! #
; ########################### DO YOU? ##############################
; ##################################################################

bg_dir_3 = bgdir + strtrim('-e') +strtrim(string(emin, F='(I04)')) + strtrim('-') + strtrim(string(emax, F='(I04)'))

journal,'background_model.log'

restore,'/afs/ipp-garching.mpg.de/mpe/gamma/instruments/integral/shared_analysis/cookbook/db_files/spi_init_saves.sav',/v

x = 18.25 + dindgen(3964)

x_idx_range2 = where(x gt emin and x lt emax)

process = 1
clobber = 1
print = 1
tscw_build_bg_multirange_rev_ebds_fl, spidir=spidir,bg_dir=bgdir,tracer=tracer, x_idx_range2=x_idx_range2,scw_file=scw_file,process=process,clobber=clobber,print=print

spawn,'mv background_model.log '+bg_dir_3

exit
