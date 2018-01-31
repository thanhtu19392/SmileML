library('acepack')
library('stats')

compute_pairwise_ace <- function(df, num_features, cat_features) {
    for (col in cat_features) {
        df[, col] = as.numeric(df[, col])
    }
    df = df[, c(cat_features, num_features)]
    n = ncol(df)
    n_cat = length(cat_features)
    result = c()
    for (i in seq(1, n - 1)) {
        y = df[, i]
        x = df[, (i + 1):n]
        cat_indexes = seq(1, n_cat - i + 1, length = max(0, n_cat - i + 1))
        a = ace(x, y, cat = cat_indexes)
        acescores = cor(a$tx, a$ty)
        result = c(result, rep(0, i-1), 1, acescores)
    }
    result = c(result, rep(0, n-1), 1)
    result = matrix(result, n, n)
    result = result + t(result) - diag(1, n, n)
    return (result)
}
