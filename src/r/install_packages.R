install.packages("readr", repos = "https://cloud.r-project.org")

if (!requireNamespace("readr", quietly = TRUE)) {
  stop("readr installation failed.")
}