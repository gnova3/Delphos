# Check if remotes is installed (it should be via conda)
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", repos = "https://cloud.r-project.org/")
}

# Install the exact version of Apollo
cat("Installing Apollo 0.3.7...\n")
remotes::install_version(
  "apollo", 
  version = "0.3.7", 
  repos = "https://cloud.r-project.org/",
  upgrade = "never"
)
cat("Apollo 0.3.7 installed successfully.\n")
