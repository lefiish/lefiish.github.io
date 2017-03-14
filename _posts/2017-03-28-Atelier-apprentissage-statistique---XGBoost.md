Cet atelier a pour objectif de montrer l'utilisation du package `xgboost` pour créer des modèles de classification.

1 Importation des données
=========================

Installer et charger le package `data.table` pour charger rapidement les données dans R (les autres packages sont très souvent moins rapides).
Ce *dataset* concerne des campagnes de marketing direct afin de détecter si un client d'une banque va effectuer un dépôt à terme. Les données peuvent être téléchargées [à cette adresse](https://archive.ics.uci.edu/ml/machine-learning-databases/00222/) : récupérer le fichier *bank-additional-full.csv* du fichier *zip* *bank-additional.zip*.

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

La méthode **XGBoost** ne sait pas gérer les variables caractères. Il faut donc les modifier, par exemple en utilisant l'encodage *one-hot* : pour une variable initiale donnée, nous créons autant de variables qu'il y a de modalités. Ces variables sont des indicatrices de chaque modalité.

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
for(i in 1:10){
  set.seed(1234)
  data[, paste0('bruit', i)] <- sample.int(i+1, size = nrow(data), replace = TRUE)
}
```

3 Découpage des données
=======================

Nous découpons nos données en 2 échantillons : l'apprentissage, qui nous servira à construire les modèles, et le test, à évaluer les performances du modèle. Nous gardons 70% des données pour l'apprentissage, 30% pour le test.
Il faudra évaluer les performances qu'une seule fois sur le test afin d'éviter tout risque de *sur-apprentissage*. Il est par exemple proscrit de valider un modèle, évaluer ses performances sur l'échantillon test, et revenir sur au départ pour modifier un paramètre ou changer de famille de modèles.

``` r
set.seed(123)
indices <- as.logical(rbinom(nrow(data), 1, 0.7))
app <- data[indices, ]
test <- data[!indices, ]
y_app <- y[indices]
y_test <- y[!indices]
```

4 Modèles **XGBoost**
=====================

Afin d'optimiser les traitements, le package `xgboost` utilise son propre type de stockage, les `DMatrix`. Le package contient une fonction pour convertir un `data.frame` en `DMatrix`.

``` r
donnee <- xgb.DMatrix(data = data.matrix(app), label = y_app)
class(donnee)
```

    ## [1] "xgb.DMatrix"

4.1 Premier modèle et *sur-apprentissage*
-----------------------------------------

Entraînons notre premier modèle **XGBoost** sur les données d'apprentissage.
C'est un modèle de classification binaire, donc l'objectif doit être `reg:logistic`. Notons que le mot-clé `logistic` ne signifie pas que c'est une régression logistique ; voir [cette page](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md#learning-task-parameters) pour choisir le bon objectif.
Nous utilisons l'aire sous la courbe ROC pour évaluer nos performances.
Nous choisissons de faire 50 itérations.
Si possible, les 4 coeurs de la machine seront utilisés pour paralléliser les traitements.
Les autres paramètres par défaut sont utilisés.

``` r
xgb <- xgboost(data = donnee,
               nrounds = 50,
               objective = 'reg:logistic', eval_metric = 'auc', nthread = 4, verbose = 1)
```

    ## [1]  train-auc:0.937084 
    ## [2]  train-auc:0.940098 
    ## [3]  train-auc:0.942914 
    ## [4]  train-auc:0.948871 
    ## [5]  train-auc:0.950330 
    ## [6]  train-auc:0.953222 
    ## [7]  train-auc:0.955165 
    ## [8]  train-auc:0.956548 
    ## [9]  train-auc:0.958162 
    ## [10] train-auc:0.959587 
    ## [11] train-auc:0.960613 
    ## [12] train-auc:0.961170 
    ## [13] train-auc:0.962051 
    ## [14] train-auc:0.962482 
    ## [15] train-auc:0.963356 
    ## ...
    ## [45] train-auc:0.975080 
    ## [46] train-auc:0.975587 
    ## [47] train-auc:0.976019 
    ## [48] train-auc:0.976494 
    ## [49] train-auc:0.976680 
    ## [50] train-auc:0.976970

``` r
ggplot(xgb$evaluation_log, aes(x=iter, y=train_auc)) +
  geom_line() +
  xlab('Nombre d\'itérations') +
  ylab('AUC échantillon d\'apprentissage')
```

![](Atelier_files/figure-markdown_github/unnamed-chunk-11-1.png)

L'AUC semble augmenter au fur et à mesure que le nombre d'itérations augmente. *Question 1* : est-il possible d'obtenir un AUC égal à 1 ?
Pour rappel, voici les principaux paramètres et leur impact sur les performances : "positif" signifie que les performances augmentent si le paramètre prend des valeurs plus élevées.

| Paramètre    | Impact sur les performances |
|--------------|-----------------------------|
| *nround*     | Positif                     |
| *eta*        | Positif                     |
| *max\_depth* | Positif                     |

![](Atelier_files/figure-markdown_github/unnamed-chunk-12-1.png)
