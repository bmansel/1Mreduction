pyinstaller /home/brad/Dropbox/NSRRC_postdoc/code/weiComplex/weiComplex/weiComplex.py --noconfirm --hidden-import="fabio.adscimage" --hidden-import="fabio.mpaimage" --hidden-import="fabio.pixiimage" --hidden-import="fabio.binaryimage" --hidden-import="fabio.HiPiCimage" --hidden-import="fabio.pilatusimage" --hidden-import='PIL._tkinter_finder' --onedir --consol 

cp site-packages/hdf5plugin linux_build/dist/weiComplex
