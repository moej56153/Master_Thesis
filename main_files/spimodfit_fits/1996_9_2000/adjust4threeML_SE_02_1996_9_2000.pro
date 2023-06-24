response_file = '/home/jmoeller/cookbook/SPI_cookbook/MT/Crab_fits/fit_Crab_SE_02_1996_9_2000/spectral_response.rmf.fits'

spectrum_file_01 = '/home/jmoeller/cookbook/SPI_cookbook/MT/Crab_fits/fit_Crab_SE_02_1996_9_2000/spectra_Crab_Nebula.fits'

spectrum_file_02 = '/home/jmoeller/cookbook/SPI_cookbook/MT/Crab_fits/fit_Crab_SE_02_1996_9_2000/spectra_A0535+26a.fits'

;;;;;;;;;;;;;;;;;;;;;
;; RESPONSE MATRIX ;;
;;;;;;;;;;;;;;;;;;;;;

; second extension
h2 = headfits(response_file,exten=2)
sxaddpar,h2,'EXTNAME','MATRIX'
modfits,response_file,0,h2,exten_no=2

; third extension
h3 = headfits(response_file,exten=3)
sxaddpar,h3,'EXTNAME','EBOUNDS'
modfits,response_file,0,h3,exten_no=3



;;;;;;;;;;;;;;;;;;;;
;; SPECTRUM FILES ;;
;;;;;;;;;;;;;;;;;;;;

; spectrum file_01
hs_01 = headfits(spectrum_file_01,exten=2)
sxaddpar,hs_01,'EXTNAME','SPECTRUM'
modfits,spectrum_file_01,0,hs_01,exten_no=2

; spectrum file_02
hs_02 = headfits(spectrum_file_02,exten=2)
sxaddpar,hs_02,'EXTNAME','SPECTRUM'
modfits,spectrum_file_02,0,hs_02,exten_no=2


exit
