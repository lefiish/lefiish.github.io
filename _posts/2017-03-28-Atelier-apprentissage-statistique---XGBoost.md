Cet atelier a pour objectif de montrer l'utilisation du package `xgboost` pour créer des modèles de classification.

Commençons par charger les libraires nécessaires.

``` r
setwd('C:/Users/thipoissonnier/Dropbox/Public/Avisia/Formation app stat')

library(Ckmeans.1d.dp)
library(cvAUC)
library(data.table)
library(dplyr)
library(ggplot2)
library(pROC)
library(ROCR)
library(xgboost)
```

1 Importation des données
=========================

Le package `data.table` sert à charger rapidement les données dans R ; les autres packages sont très souvent moins rapides.
Ce *dataset* concerne des campagnes de marketing direct et a pour but de détecter si un client d'une banque va effectuer un dépôt à terme. Les données peuvent être téléchargées [à cette adresse](https://archive.ics.uci.edu/ml/machine-learning-databases/00222/) (fichier *bank-additional-full.csv* de l'archive *bank-additional.zip*).

Nous avons déjà récupéré les données, recodé la cible de *yes*/*no* en 1/0 (afin que les algorithmes fonctionnent).

``` r
data <- fread('bank-additional-full-atelier.csv', sep=';', data.table = FALSE)
```

<!--
data$y <- (data$y == 'yes') %>% as.numeric

Rajoutons quelques variables bruitées, c'est-à-dire indépendantes de la cible à modéliser, afin de voir si nos algorithmes sauront ne pas faire de __sur-apprentissage__ sur ces variables.

for(i in 1:10){
  set.seed(1234)
  data[, paste0('var', i)] <- sample.int(i+1, size = nrow(data), replace = TRUE)
}

write.csv(data, file='bank-additional-full-atelier.csv', sep = ';', row.names = F)
-->
2 Traitement des variables
==========================

Nous séparons les *features* et la cible afin de simplifier nos codes ultérieurs.

``` r
y <- data$y
data$y <- NULL
```

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

``` r
data <- data[, !(colnames(data) %in% char)]
```

3 Découpage des données
=======================

Nous découpons nos données en 2 échantillons : l'échantillon d'apprentissage, qui servira à construire les modèles, et celui de test, qui servira à évaluer les performances du modèle. Nous gardons 70% des données pour l'apprentissage, 30% pour le test.
Il faudra évaluer les performances qu'une seule fois sur le test afin d'éviter tout risque de **sur-apprentissage**. Il est par exemple proscrit de calculer un modèle, évaluer ses performances sur l'échantillon test, et revenir sur au départ pour modifier un paramètre ou changer de famille de modèle.

``` r
set.seed(123)
indices <- as.logical(rbinom(nrow(data), 1, 0.7))
app <- data[indices, ]
test <- data[!indices, ]
y_app <- y[indices]
y_test <- y[!indices]
```

4 Modèles XGBoost
=================

Afin d'optimiser les traitements, le package `xgboost` utilise son propre type de stockage, les `DMatrix`. Le package contient une fonction pour convertir un `data.frame` en `DMatrix`.

``` r
donnee <- xgb.DMatrix(data = data.matrix(app), label = y_app)
```

4.1 Premier modèle et sur-apprentissage
---------------------------------------

Entraînons notre premier modèle **XGBoost** sur les données d'apprentissage. Voici les paramètres que nous avons renseignés :

-   C'est un modèle de classification binaire, donc l'objectif doit être `reg:logistic`. Notons que le mot-clé `logistic` ne signifie pas que c'est une régression logistique ; voir [cette page](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md#learning-task-parameters) pour choisir le bon objectif.
-   Nous utilisons l'aire sous la courbe ROC pour évaluer nos performances.
-   Nous choisissons de faire 50 itérations.
-   Si possible, les 4 coeurs de la machine seront utilisés pour paralléliser les traitements.
-   Pour les autres paramètres, les valeurs par défaut sont utilisées.

Un code pour évaluer un modèle **XGBoost** simple et tracer l'AUC en fonction du nombre d'itérations est donné ci-dessous.

``` r
# Modèle XGBoost
xgb <- xgboost(data = donnee,
               nrounds = 50,
               eta = 0.3, max_depth = 6, # valeurs par défaut
               objective = 'reg:logistic', eval_metric = 'auc', nthread = 4, 
               verbose = 1, print_every_n = 10)
```

    ## [1]  train-auc:0.936947 
    ## [11] train-auc:0.961097 
    ## [21] train-auc:0.966243 
    ## [31] train-auc:0.970659 
    ## [41] train-auc:0.974235 
    ## [50] train-auc:0.978850

``` r
# AUC en fonction du nombre d'itérations
ggplot(xgb$evaluation_log, aes(x=iter, y=train_auc)) +
  geom_line() +
  xlab('Nombre d\'itérations') +
  ylab('AUC échantillon d\'apprentissage')
```

![](assets/unnamed-chunk-8-1.png)

L'AUC semble augmenter au fur et à mesure que le nombre d'itérations augmente.
***Question 1*** : est-il possible d'obtenir un AUC égal à 1 ?
Pour rappel, voici les principaux paramètres et leur impact sur les performances : "positif" signifie que les performances augmentent si le paramètre prend des valeurs plus élevées.

| Paramètre    | Impact sur les performances |
|--------------|-----------------------------|
| *eta*        | Positif                     |
| *max\_depth* | Positif                     |
| *nround*     | Positif                     |

Le résultat doit être similaire à cette courbe :

    ## [1]  train-auc:0.957396 
    ## [11] train-auc:0.997027 
    ## [21] train-auc:0.999779 
    ## [31] train-auc:0.999999 
    ## [35] train-auc:1.000000

![](assets/unnamed-chunk-9-1.png)

4.2 Validation croisée
----------------------

Le modèle a réussi à coller parfaitement à nos données d'apprentissage, mais on ne sait pas quelle sera la performance sur de nouvelles données, par exemple nos données de test.
Voici une manière de calculer les performances sur un nouveau jeu de données (si le dernier modèle obtenu a été nommé `xgb2`) :

``` r
# Application du modèle : récupération des probabilités
preds <- predict(xgb2, data.matrix(test))

# Calcul de l'AUC sur l'échantillon test
AUC(predictions = preds, labels = y_test)
```

***Question 2*** : quel est l'AUC sur l'échantillon test ? Est-il supérieur à l'AUC sur l'échantillon d'apprentissage ? à peu près égal ? inférieur ? très inférieur ? Quel phénomène est mis en évidence ici ?

Afin d'entraîner un modèle sur des données et de vérifier en même temps ses performances sur un autre jeu de données, il est possible d'utiliser le paramètre `watchlist`. Cela revient à faire de la validation *hold-out*.

``` r
watchlist <- list(train = donnee, 
                  test = xgb.DMatrix(data = data.matrix(test), label = y_test))

xgb.train(data = donnee,
          nrounds = 50,
          objective = 'reg:logistic', eval_metric = 'auc', nthread = 4,
          verbose = 1, print_every_n = 5, watchlist = watchlist)
```

Une autre manière de faire, fortement recommandée, est d'utiliser la validation croisée *k*-fois. Pour la validation croisée 5 fois, cela consiste à (cf. image ci-dessous) :

1.  Diviser les données d'apprentissage en *k* = 5 sous-parties distinctes,
2.  Répéter 5 fois le même calcul :
    1.  Entraîner le modèle avec les données de 4 sous-parties,
    2.  Appliquer le modèle sur la dernière sous-partie et calculer ses performances

3.  Faire la moyenne des 5 indicateurs de performance obtenus.

![](5fold.png)

 

Cette méthode possède l'avantage de pouvoir estimer de manière efficace la performance de notre modèle sur un nouveau jeu de données, sans utiliser les données de test, et en utilisant toutes les données d'apprentissage. Notons que si le jeu de données à disposition est très volumineux et possède beaucoup de colonnes, il peut être coûteux de faire de la validation croisée. Dans ce cas-là, une validation *hold-out* peut suffire.

La validation croisée se traduit dans le package `xgboost` de cette façon :

``` r
xgb_cv <- xgb.cv(data = donnee,
                 nrounds = 50, 
                 objective = 'reg:logistic', eval_metric = 'auc', nthread = 4,
                 verbose = 1, print_every_n = 10,
                 nfold = 5)
```

***Question 3*** : à l'aide du code graphique suivant, tracer l'évolution de l'AUC en fonction du nombre d'itérations, sur les données qui ont servi à construire les modèles et sur les données "protégées".
Quelle est la forme des 2 courbes ? Qu'en déduit-on sur le paramètre *nround* ? En quoi est-ce différent des forêts aléatoires ?

``` r
ggplot(xgb_cv$evaluation_log, aes(x = iter)) + 
  geom_line(aes(y = train_auc_mean, colour = 'Apprentissage')) + 
  geom_line(aes(y = test_auc_mean, colour = 'Validation croisée')) +
  scale_colour_manual('', breaks = c('Apprentissage', 'Validation croisée'), values = c('blue', 'purple')) +
  xlab('Nombre d\'itérations') +
  ylab('AUC')
```

<!-- On voit bien que les performances sur les données utilisées pour le modèle sont trop optimistes : à partir d'un certain nombre d'itérations les performances mesurées par validation croisée stagnent puis diminuent légèrement. Le modèle a parfaitement collé aux données mais a appris des règles qui ne se généralisent pas à de nouvelles données. -->
4.3 Optimisation des hyper-paramètres
-------------------------------------

Nous avons vu que choisir `eta` proche de 1 et `max_depth` trop élevé apporte du **sur-apprentissage**. La solution est de baisser ces deux paramètres.
Nous proposons une méthode (il en existe d'autres) pour optimiser les performances de l'algorithme :

1.  Définir une grille de valeurs pour les paramètres `max_depth` et `eta`,
2.  Choisir un nombre d'itérations assez grand pour que les performances en validation croisée convergent,
3.  Boucler sur `max_depth` et `eta` :
    1.  Evaluer le modèle avec les valeurs des paramètres `max_depth` et `eta`,
    2.  Récupérer le nombre d'itérations qui donne l'AUC optimal et la valeur de l'AUC,

4.  Choisir `max_depth` et `eta` qui donnent le meilleur AUC au global.

Voici la transcription en code :

``` r
tuning <- data.frame(depth = c(4, 6, 8) %>% rep(4),
                     eta = c(0.05, 0.1, 0.15, 0.2) %>% rep(each = 3),
                     auc_optim = numeric(12),
                     iter_optim = integer(12))

for(i in seq(12)){
  print(paste('Itération', i, 'sur 12 : max_depth =', tuning$depth[i], 'et eta =', tuning$eta[i]))
  xgb_temp <- xgb.cv(data = donnee,
                     nrounds = 250, eta = tuning$eta[i], max_depth = tuning$depth[i],
                     objective = 'reg:logistic', eval_metric = 'auc', nthread = 4,
                     verbose = 1, print_every_n = 25,
                     nfold = 5)
  
  tuning$auc_optim[i] <- xgb_temp$evaluation_log %>%
    select(test_auc_mean) %>%
    max
  
  tuning$iter_optim[i] <- xgb_temp$evaluation_log %>%
    filter(test_auc_mean == max(test_auc_mean)) %>%
    select(iter)
}

ggplot(tuning, aes(x = depth, y = eta, fill = auc_optim, label = auc_optim)) +
  geom_tile() +
  geom_text(aes(label = auc_optim %>% round(4)), color = 'white') +
  labs(title = 'Tuning de eta et max_depth')
```

Et la graphique obtenu est :

![](assets/gridsearch.png)

 

Les meilleures performances sont obtenues pour `max_depth` =6 et `eta` =0.1, après 95 itérations (en faisant tourner le code). Pour les futures évaluations du modèle, il sera préférable de choisir un nombre d'itérations un peu supérieur à 100 pour s'assurer de la convergence des performances.

***Question 4*** : les performances des modèles pour `max_depth` égal à 4 ou 6 et `eta` entre 0.1 et 0.2 sont-elles assez proches ? Que pourrait-on en déduire sur le temps à passer à optimiser les paramètres du modèle ? Peut-on faire un parallèle avec les forêts aléatoires ?

Notons qu'il n'existe pas de combinaison optimale des hyper-paramètres qui fonctionnent sur tout type de données. Sur des données avec des interactions complexes, les arbres doivent être suffisamment profonds. Sur des données avec des valeurs aberrantes ou extrêmes, le *learning rate* ne devra pas être trop élevé. Sur de gros jeux de données, un *learning rate* trop faible ou des arbres trop peu profonds augmenteront le temps de calcul.

***Question 5*** : optimiser de la même façon les paramètres `min_child_weight`, `colsample_by_tree` et `subsample`.
Quelle est la meilleure combinaison ?

***Question 6*** : tracer le graphique de performance du meilleur modèle et obtenir l'AUC maximal (par rapport au nombre d'itérations).
La valeur de l'AUC est-elle la même que lors de l'optimisation des hyperparamètres ? Pourquoi ?

4.4 Comparaison finale des performances
---------------------------------------

***Question 7*** : comparer l'AUC calculé sur l'échantillon de test pour le 1<sup>er</sup> modèle obtenu (valeurs par défauts des paramètres), pour le modèle qui a sur-appris les données, et pour le "meilleur" modèle. Que constatons-nous ?

 

Le code suivant sert à tracer les courbes ROC du modèle sur-appris et du meilleur modèle.

``` r
# Courbes ROC
plot.roc(y_test, preds, col = 'blue')
plot.roc(y_test, preds_best, print.auc = T, print.auc.y = 0.5, xlim = c(1,0), col = 'black', add = TRUE)
```

![](assets/unnamed-chunk-19-1.png)

5 Importance des variables
==========================

***Question 8*** : tracer l'importance des 15 1<sup>eres</sup> variables grâce au code suivant. Quelles sont les variables les plus importantes ?

Les différents types d'importance sont :

-   *Gain* : la moyenne du gain en précision quand une variable est utilisée dans les arbres,
-   *Cover* : le nombre d'observations concernées par les coupures liées à la variable,
-   *Frequency* : nombre fois qu'une variable est utilisée pour une coupure.

``` r
# Calcul de l'importance des variables
importance_matrix <- xgb.importance(colnames(data), model = xgb_best)

# Tracé graphique
xgb.ggplot.importance(importance_matrix, top_n = 15)
```

***Question 9*** : les variables commençant par "var" ont été rajoutées et ne sont pas présentes dans le jeu de données initial. Elles correspondent à des variables aléatoires donc théoriquement indépendantes de la cible. Vérifiez que leur importance est suffisamment faible.

Le code suivant permet de calculer les rangs pour chaque variable d'importance, et de sélectionner les importances pour les variables commençant par "var".

``` r
importance_matrix <- importance_matrix %>%
  mutate(rank_gain = Gain %>% desc %>% dense_rank) %>%
  mutate(rank_cover = Cover %>% desc %>% dense_rank) %>%
  mutate(rank_freq = Frequency %>% desc %>% dense_rank)

importance_matrix %>% subset(
  (importance_matrix %>%
     select(Feature) %>%
     sapply(. %>% substr(1, 3)) == 'var'
   ) %>% as.vector
  )
```

6 Bonus
=======

6.1 Fonctions d'entraînement
----------------------------

Les fonctions `xgboost` et `xgb.train` sont quasiment interchangeables, la 2<sup>e</sup> étant un peu plus riche que la 1<sup>e</sup> au niveau des paramètres.

6.2 *Early stopping*, sélection du nombre d'itérations
------------------------------------------------------

Le paramètre `early_stopping_rounds` =*k* de la fonction `xgboost` sert à arrêter les itérations si les performances n'augmentent pas consécutivement pendant *k* itérations. Il sert à arrêter l'algorithme s'il converge rapidement par rapport au nombre d'itérations qui a été spécifié dans le code.

Une fois le modèle entraîné, le paramètre `ntreelimit` de la fonction `predict`, qui sert à scorer de nouvelles données, permet de spécifier le nombre d'arbres qui seront utilisées pour l'application. C'est utile quand le nombre d'itérations a été choisi trop élevé.

Caret

Pipelearner

Scorer des nouvelles données avec la sortie d'un xgb.cv ?

Autres fonctions du package
