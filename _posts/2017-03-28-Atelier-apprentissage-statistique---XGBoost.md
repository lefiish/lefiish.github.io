1 Importation des données
=========================

Installer le package `data.table` pour charger rapidement les données dans R (les autres packages sont très souvent moins rapides). Ce *dataset* concerne des campagnes de marketing direct afin de détecter si un client d'une banque va effectuer un dépôt à terme.

``` r
data <- fread('bank-additional-full.csv', sep=';', data.table = FALSE)
```

Récupérer la cible dans un vecteur à part, et changer *yes*/*no* en 1/0.

``` r
y <- data$y
data$y <- NULL

y <- (y == 'yes') %>% as.numeric
```

2 Traitement des variables
==========================

La méthode **XGBoost** ne sait pas gérer les variables caractères. Il faut donc les modifier, par exemple en utilisant l'encodage *one-hot* : pour une variable initiale donnée, on crée autant de variables qu'il y a de modalités. Ces variables sont des indicatrices de chaque modalité.

``` r
classes <- data[1, ] %>% sapply(class)
char <- (classes == 'character') %>% which %>% names

for(j in char){
  print(j)
  for(i in data[, j] %>% unique){
    # print(i)
    data[, paste0(j, '_', i)] <- data[, j] == i
  }
}
```

    ## [1] "job"
    ## [1] "marital"
    ## [1] "education"
    ## [1] "default"
    ## [1] "housing"
    ## [1] "loan"
    ## [1] "contact"
    ## [1] "month"
    ## [1] "day_of_week"
    ## [1] "poutcome"
