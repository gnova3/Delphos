required_r_major <- 4
required_r_minor <- 4

current <- getRversion()
if (current < sprintf("%d.%d.0", required_r_major, required_r_minor)) {
  stop(
    sprintf(
      "Delphos requires R >= %d.%d.0. Current R version is %s.",
      required_r_major,
      required_r_minor,
      as.character(current)
    )
  )
}

install_if_missing <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    install.packages(package, repos = "https://cloud.r-project.org")
  }
}

install_if_missing("remotes")

install_exact <- function(package, version) {
  installed <- requireNamespace(package, quietly = TRUE)
  if (installed && as.character(utils::packageVersion(package)) == version) {
    message(sprintf("%s %s is already installed.", package, version))
    return(invisible(TRUE))
  }

  remotes::install_version(
    package,
    version = version,
    repos = "https://cloud.r-project.org",
    upgrade = "never"
  )
}

install_exact("apollo", "0.3.7")

message("Delphos R requirements are installed.")
