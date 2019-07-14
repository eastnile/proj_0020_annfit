# Set global file paths
try(detach(init), silent = T)

init = new.env()
init$findgdrivepath = function() {
    systemname = Sys.info()["nodename"]
    switch(systemname,
          'SIDDHARTHA' = { gdrivepath = "C:/Users/zhaochenhe/Google Drive/" },
          'IMMANUEL-PC' = { gdrivepath = "C:/Users/immanuel/Google Drive/" },
          { # Else
            print("Unable to identify computer.")
            gdrivepath = 'unknown'
          }
    )
    print(paste("Google drive path set to", gdrivepath))
    p = list()
    p$gdrivepath = gdrivepath
    assign('p', p, pos = globalenv())
}


# Load commonly used functions

# Install a list of packages (string array) if they aren't installed already and load them.
init$installif = function(liblist) {
    new.packages = liblist[!(liblist %in% installed.packages()[, 'Package'])]
    if (length(new.packages)) { install.packages(new.packages) }
    lapply(liblist, require, character.only = TRUE)
    }

# Load a string array of packages
init$lib = function(liblist) {
    lapply(liblist, require, character.only = TRUE)
}

# System Variables
options(max.print = 1000)
Sys.setenv(JAVA_HOME = 'C:\\Program Files\\Java\\jdk-10.0.1')
local({
    r <- getOption("repos")
    r["CRAN"] <- "http://cloud.r-project.org/"
    options(repos = r)
})

# Final steps
attach(init)
findgdrivepath()
source('proj.R')