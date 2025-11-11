default : attrs install

attrs :
	Rscript -e "library(Rcpp) ; compileAttributes(verbose=TRUE)"

build : attrs clean
	R CMD build . --resave-data
	R CMD Rd2pdf -o doc.pdf .

check : build
	R CMD check emIRTEDM*.tar.gz

fullinstall : build
	R CMD INSTALL emIRTEDM*.tar.gz

install : attrs clean
	R CMD INSTALL .

remove :
	R CMD REMOVE emIRTEDM

clean :
	rm -f emIRTEDM*.tar.gz
	rm -fr emIRTEDM.Rcheck
	rm -f ./src/*.o
	rm -f ./src/*.so
	rm -f ./src/*.rds
	rm -f ./inst/lib/*
	rm -f ./doc.pdf
