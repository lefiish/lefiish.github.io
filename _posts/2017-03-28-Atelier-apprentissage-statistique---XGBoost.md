Cet atelier a pour objectif de montrer l'utilisation du package `xgboost` pour créer des modèles de classification.

1 Importation des données
=========================

Installer et charger le package `data.table` pour charger rapidement les données dans R (les autres packages sont très souvent moins rapides). Ce *dataset* concerne des campagnes de marketing direct afin de détecter si un client d'une banque va effectuer un dépôt à terme. Les données peuvent être téléchargées [à cette adresse](https://archive.ics.uci.edu/ml/machine-learning-databases/00222/) : récupérer le fichier *bank-additional-full.csv* du fichier *zip* *bank-additional.zip*.

``` r
data <- fread('bank-additional-full.csv', sep=';', data.table = FALSE)
```

Récupérer la cible dans un vecteur à part, et changer *yes*/*no* en 1/0 afin que les algorithmes utilisés puissent fonctionner.

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
  print(paste0('Variable : ', j))
  for(i in data[, j] %>% unique){
    # print(i)
    data[, paste0(j, '_', i)] <- data[, j] == i
  }
}
```

    ## [1] "Variable : job"
    ## [1] "Variable : marital"
    ## [1] "Variable : education"
    ## [1] "Variable : default"
    ## [1] "Variable : housing"
    ## [1] "Variable : loan"
    ## [1] "Variable : contact"
    ## [1] "Variable : month"
    ## [1] "Variable : day_of_week"
    ## [1] "Variable : poutcome"

Il est aussi possible de recoder certaines variables caractère de manière ordinale : ce sont les variables qui peuvent être définies comme un ordre. Par exemple, bon = 1, moyen = 2, mauvais = 3.

``` r
data <- data %>%
  mutate(education2 = revalue(education, c('illiterate' = 0, 'basic.4y' = 1, 'basic.6y' = 2, 'basic.9y' = 3, 'high.school' = 4,
                                                'professional.course' = 5, 'university.degree' = 6, 'unknown' = NA)) %>% as.numeric) %>%
  mutate(month2 = revalue(month, c('mar'=3, 'apr'=4, 'may'=5, 'jun'=6, 'jul'=7, 'aug'=8, 'sep'=9, 'oct'=10, 'nov'=11, 'dec'=12)) %>% as.numeric) %>%
  mutate(day2 = revalue(day_of_week, c('mon'=1, 'tue'=2, 'wed'=3, 'thu'=4, 'fri'=5)) %>% as.numeric) %>%
  mutate(poutcome2 = revalue(poutcome, c('failure'=0, 'nonexistent'=1, 'success'=2)) %>% as.numeric)

data <- data[, !(colnames(data) %in% char)]
```

Rajoutons quelques variables bruitées, c'est-à-dire indépendantes de la cible à modéliser, afin de voir si nos algorithmes sauront ne pas faire d'*overfitting* sur ces variables.

``` r
for(i in 1:10)
  data[, paste0('bruit', i)] <- sample.int(i+1, size = nrow(data), replace = TRUE)
```
