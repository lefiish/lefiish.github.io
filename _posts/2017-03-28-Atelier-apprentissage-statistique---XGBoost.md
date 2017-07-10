Cet atelier a pour objectif de montrer l'utilisation du package `xgboost` pour créer des modèles de classification.

Commençons par charger les libraires nécessaires.

``` r
setwd('C:/Users/thipoissonnier/Dropbox/Public/Avisia/Formation app stat')

library(Ckmeans.1d.dp)    # pour effectuer le clustering des importances des variables
library(cvAUC)            # pour calculer l'AUC rapidement
library(data.table)       # pour importer les données et offrir des manipulations de données simples
library(dplyr)            # pour manipuler concrètement des données, notamment avec l'opérateur %>%
library(ggplot2)          # pour faire de beaux graphiques
library(pROC)             # pour tracer la courbe ROC rapidement
library(xgboost)          # le package XGBoost
```

1 Importation des données
=========================

Le *package* `data.table` sert à charger rapidement les données dans R ; les autres packages sont très souvent moins rapides. Le paramètre `data.table` = FALSE force à stocker les données sous forme d'un `data.frame` au lieu d'un `data.table`, le 1<sup>er</sup> étant + simple à manipuler. Ce *dataset* concerne des campagnes de marketing direct et a pour but de détecter si un client d'une banque va effectuer un dépôt à terme. Les données peuvent être téléchargées [à cette adresse](https://archive.ics.uci.edu/ml/machine-learning-databases/00222/) (fichier *bank-additional-full.csv* de l'archive *bank-additional.zip*).

Nous avons déjà récupéré les données, recodé la cible de *yes*/*no* en 1/0 (afin que les algorithmes fonctionnent).

``` r
# Import des données
data <- fread('bank-additional-full-atelier.csv', sep=';', data.table = FALSE)
```

2 Traitement des variables
==========================

Nous séparons les *features* et la cible afin de simplifier nos codes ultérieurs.

``` r
y <- data$y
data$y <- NULL
```

La méthode **XGBoost** ne sait pas gérer les variables caractères. Il faut donc les modifier, par exemple en utilisant l'encodage *one-hot* : pour une variable initiale donnée, nous créons autant de variables qu'il y a de modalités. Ces variables sont des indicatrices de chaque modalité.

``` r
# Récupération des noms des variables caractère
classes <- data[1, ] %>% sapply(class)
char <- (classes == 'character') %>% which %>% names

# Binarisation
for(j in char){
  for(i in data[, j] %>% unique){
    data[, paste0(j, '_', i)] <- data[, j] == i
  }
}

# Suppression des variables caractère
data <- data[, !(colnames(data) %in% char)]
```

3 Découpage des données
=======================

Nous découpons nos données en 2 échantillons : l'échantillon d'apprentissage, qui servira à construire les modèles, et celui de test, qui servira à évaluer les performances du modèle. Nous gardons 70% des données pour l'apprentissage, 30% pour le test.
Il faudra évaluer les performances qu'une seule fois sur le test afin d'éviter tout risque de **sur-apprentissage**. Il est par exemple proscrit de calculer un modèle, évaluer ses performances sur l'échantillon test, et revenir sur ce modèle pour modifier un paramètre ou changer de famille de modèle.

``` r
# Pour garder le même split
set.seed(123)

# Split
indices <- (data %>% nrow %>% rbinom(1, 0.7)) %>% as.logical
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
  # le résultat du modèle contient l'objet evaluation_log, 
  # qui récupère les performances à chaque itération.
ggplot(xgb$evaluation_log, aes(x=iter, y=train_auc)) +
  geom_line() +
  xlab('Nombre itérations') +
  ylab('AUC échantillon apprentissage')
```

![](/assets/unnamed-chunk-8-1.png)

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
    ## [35] train-auc:1.000000

![](/assets/unnamed-chunk-9-1.png)

4.2 Validation croisée
----------------------

Le modèle a réussi à coller parfaitement à nos données d'apprentissage, mais on ne sait pas quelle sera la performance sur de nouvelles données, par exemple nos données de test.
Voici une manière de calculer les performances sur un nouveau jeu de données (si le dernier modèle obtenu a été nommé `xgb2`) :

``` r
# Application du modèle : récupération des probabilités
preds <- predict(xgb2, data.matrix(test))

# Calcul de l'AUC sur l'échantillon test
auc_overfit <- AUC(predictions = preds, labels = y_test)
```

***Question 2*** : quel est l'AUC sur l'échantillon test ? Est-il supérieur à l'AUC sur l'échantillon d'apprentissage ? à peu près égal ? inférieur ? très inférieur ? Quel phénomène est mis en évidence ici ?

Afin d'entraîner un modèle sur des données et de vérifier en même temps ses performances sur un autre jeu de données, il est possible d'utiliser le paramètre `watchlist`. Cela revient à faire de la validation *hold-out*.

``` r
# Ce n'est pas nécessaire de lancer les 2 instructions suivantes :
watchlist <- list(train = donnee, 
                  test = xgb.DMatrix(data = data.matrix(test), label = y_test))

xgb.train(data = donnee,
          nrounds = 50,
          objective = 'reg:logistic', eval_metric = 'auc', nthread = 4,
          verbose = 1, print_every_n = 5, watchlist = watchlist)
```

Une autre manière de faire, fortement recommandée, est d'utiliser la validation croisée *k*-fois, ou *cross-validation*. Pour la validation croisée 5 fois, cela consiste à (cf. image ci-dessous) :

1.  Diviser les données d'apprentissage en *k* = 5 sous-parties distinctes,
2.  Répéter 5 fois le même calcul :
    1.  Entraîner le modèle avec les données de 4 sous-parties (données *train*),
    2.  Appliquer le modèle sur la dernière sous-partie et calculer ses performances (données *test*),

3.  Faire la moyenne des 5 indicateurs de performance obtenus.

![](/assets/5fold.png)

 

Cette méthode possède l'avantage de pouvoir estimer de manière efficace la performance de notre modèle sur un nouveau jeu de données, sans utiliser les données de test, et en utilisant toutes les données d'apprentissage.
*Note* : si le jeu de données à disposition est très volumineux et possède beaucoup de colonnes, il peut être coûteux de faire de la validation croisée. Dans ce cas-là, une validation *hold-out* peut suffire.

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
  scale_colour_manual('', breaks = c('Apprentissage', 'Validation croisée'), 
                      values = c('blue', 'purple')) +
  xlab('Nombre d\'itérations') +
  ylab('AUC')
```

<!-- On voit bien que les performances sur les données utilisées pour le modèle sont trop optimistes : à partir d'un certain nombre d'itérations les performances mesurées par validation croisée stagnent puis diminuent légèrement. Le modèle a parfaitement collé aux données mais a appris des règles qui ne se généralisent pas à de nouvelles données.

4.3 Optimisation des hyper-paramètres
-------------------------------------

Nous avons vu que choisir `eta` proche de 1 et `max_depth` trop élevé apporte du **sur-apprentissage**. La solution est de baisser ces deux paramètres.
Nous proposons une méthode, appelée *grid search* (il en existe d'autres), pour optimiser les performances de l'algorithme :

1.  Définir une grille de valeurs pour les paramètres `max_depth` et `eta`,
2.  Choisir un nombre d'itérations assez grand pour que les performances en validation croisée convergent,
3.  Boucler sur `max_depth` et `eta` :
    1.  Evaluer le modèle avec les valeurs des paramètres `max_depth` et `eta`,
    2.  Récupérer le nombre d'itérations qui donne l'AUC optimal et la valeur de l'AUC,

4.  Choisir `max_depth` et `eta` qui donnent le meilleur AUC au global.

Voici la transcription en code :

``` r
# Grille de valeurs
tuning <- data.frame(depth = c(4, 6, 8) %>% rep(4),
                     eta = c(0.05, 0.1, 0.15, 0.2) %>% rep(each = 3),
                     auc_optim = numeric(12),
                     iter_optim = integer(12))

# Boucle sur les valeurs
for(i in seq(12)){
  cat(paste('Itération', i, 'sur', nrow(tuning), ': max_depth =', tuning$depth[i], 'et eta =', tuning$eta[i], '\n'))
  xgb_temp <- xgb.cv(data = donnee,
                     nrounds = 250, eta = tuning$eta[i], max_depth = tuning$depth[i],
                     objective = 'reg:logistic', eval_metric = 'auc', nthread = 4,
                     verbose = 1, print_every_n = 25,
                     nfold = 5)
  
  # Récupération de l'AUC optimal
  tuning$auc_optim[i] <- xgb_temp$evaluation_log %>%
    select(test_auc_mean) %>%
    max
  
  # Récupération du nombre d'itérations optimal
  tuning$iter_optim[i] <- xgb_temp$evaluation_log %>%
    filter(test_auc_mean == max(test_auc_mean)) %>%
    select(iter)
}

ggplot(tuning, aes(x = depth, y = eta, fill = auc_optim, label = auc_optim)) +
  geom_tile() +
  geom_text(aes(label = auc_optim %>% round(4)), color = 'white') +
  labs(title = 'Tuning de eta et max_depth')
```

Et le graphique obtenu est :

![](/assets/gridsearch.png)

 

Les meilleures performances sont obtenues pour `max_depth` =6 et `eta` =0.1, après 95 itérations (les résultats peuvent varier). Pour les futures évaluations du modèle, il sera préférable de choisir un nombre d'itérations un peu supérieur à 100 pour s'assurer de la convergence des performances.

***Question 4*** : quelle relation existe-t-il entre chaque paramètre et le nombre d'itérations optimal ? est-ce fidèle à l'intuition ?

***Question 5*** : les performances des modèles pour `max_depth` égal à 4 ou 6 et `eta` entre 0.1 et 0.2 sont-elles assez proches ? Que pourrait-on en déduire sur le temps à passer à optimiser les paramètres du modèle ? Peut-on faire un parallèle avec les forêts aléatoires ?

Notons qu'il n'existe pas de combinaison optimale des hyper-paramètres qui fonctionnent sur tout type de données. Sur des données avec des interactions complexes, les arbres doivent être suffisamment profonds. Sur des données avec des valeurs aberrantes ou extrêmes, le *learning rate* ne devra pas être trop élevé. Sur de gros jeux de données, un *learning rate* trop faible ou des arbres trop peu profonds augmenteront le temps de calcul.

***Question 6*** : optimiser de la même façon les paramètres `min_child_weight`, `colsample_by_tree` et `subsample`.
Quelle est la meilleure combinaison ?

***Question 7*** : refaire tourner le "meilleur" modèle grâce à la fonction `xgb.cv` afin de tracer le graphique de suivi de l'AUC en fonction du nombre d'itérations (en prenant les hyper-paramètres trouvés à la question précédente).
La valeur de l'AUC est-elle la même que lors de l'optimisation des hyperparamètres ? Pourquoi ?

4.4 Comparaison finale des performances
---------------------------------------

***Question 8*** : refaire tourner le modèle sur toutes les données d'apprentissage, c'est-à-dire sans *cross-validation*. Comparer l'AUC calculé sur l'échantillon de test pour le 1<sup>er</sup> modèle obtenu (valeurs par défaut des paramètres), pour le modèle qui a sur-appris les données, et pour le "meilleur" modèle. Que constatons-nous ?

 

Le code suivant sert à tracer les courbes ROC du modèle sur-appris et du meilleur modèle.

``` r
# Courbes ROC
plot.roc(y_test, preds, col = 'blue')
plot.roc(y_test, preds_best, print.auc = T, print.auc.y = 0.5, xlim = c(1,0), col = 'black', add = TRUE)
```

![](/assets/unnamed-chunk-19-1.png)

5 Importance des variables
==========================

***Question 9*** : tracer l'importance des 15 meilleures variables grâce au code suivant. Quelles sont les variables les plus importantes ?

Les différents types d'importance sont :

-   *Gain* : la moyenne du gain en précision quand une variable est utilisée dans les arbres,
-   *Cover* : le nombre d'observations concernées par les coupures liées à la variable,
-   *Frequency* : le nombre fois qu'une variable est utilisée pour une coupure.

``` r
# Calcul de l'importance des variables
importance_matrix <- xgb.importance(colnames(data), model = xgb_best)

# Tracé graphique
xgb.ggplot.importance(importance_matrix, top_n = 15)
```

***Question 10*** : les variables commençant par "var" ont été rajoutées et ne sont pas présentes dans le jeu de données initial. Elles correspondent à des variables aléatoires donc théoriquement indépendantes de la cible. Vérifiez que leur importance est suffisamment faible.
Le code suivant permet de calculer les rangs pour chaque variable d'importance, et de sélectionner les importances pour les variables commençant par "var".

``` r
# Calcul des rangs
importance_matrix <- importance_matrix %>%
  mutate(rank_gain = Gain %>% desc %>% dense_rank) %>%
  mutate(rank_cover = Cover %>% desc %>% dense_rank) %>%
  mutate(rank_freq = Frequency %>% desc %>% dense_rank)

# Importance pour les variables bruitées
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

Les fonctions `xgboost` et `xgb.train` sont quasiment interchangeables : la 2<sup>e</sup> est un peu plus riche que la 1<sup>e</sup> au niveau des paramètres.

6.2 *Early stopping*, sélection du nombre d'itérations
------------------------------------------------------

Le paramètre `early_stopping_rounds` de la fonction `xgboost` (et de `xgb.cv`) sert à arrêter les itérations si les performances n'augmentent pas consécutivement pendant *k* itérations. Il sert à arrêter l'algorithme s'il converge rapidement par rapport au nombre d'itérations qui a été spécifié dans le code.
Une fois l'algorithme entraîné, ces fonctions produisent des objets `best_iteration` et `best_ntreelimit`, qui renvoient le nombre d'itérations optimal, et `best_score`, le score optimal associé (uniquement pour la fonction `xgboost`). Ces résultats sont utiles et peuvent permettre d'automatiser une chaîne de traitements.
Enfin, le paramètre `ntreelimit` de la fonction `predict` permet de spécifier le nombre d'arbres qui seront utilisées pour scorer les nouvelles données.

***Question 11*** : jouer le code qui contient le paramètre `watchlist` (en dessous de la question 2), avec le paramètre `early_stopping_rounds` =3. Quel est le nombre d'itérations optimal et le score associé ?

6.3 Affichage des arbres
------------------------

Le package `xgboost` permet de représenter graphiquement le modèle, en affichant séparément tous les modèles simples (les arbres). La fonction `xgb.plot.tree` permet de le faire et affiche tous les arbres utilisés (si `nround` est trop grand l'affichage et l'interprétation ne seront pas aisées). [Voici un exemple.](http://dmlc.ml/rstats/2016/03/10/xgboost.html)

6.4 Utilisation avec d'autres *packages*
----------------------------------------

`caret` est un formidable *package* qui regroupe un grand nombre d'outils pour faciliter la création d'un modèle d'apprentissage statistique. Un de ses avantages est de réunir dans un même endroit un très grand nombre d'algorithmes différent, dont **XGBoost**. [Voir cette page](https://topepo.github.io/caret/available-models.html) pour la liste complète des 233 algorithmes (!).
L'implémentation de la méthode *grid search* est beaucoup plus facile avec ce package, ainsi que les étapes de pré-traitement des variables, de sélection des variables, de comparaison des modèles, et de calcul d'importance des variables.
[Cet article](http://blog.revolutionanalytics.com/2016/05/using-caret-to-compare-models.html) montre comment comparer **XGBoost** et du "simple" *gradient boosting*, de manière parallélisée, et en choisissant plusieurs combinaisons d'hyper-paramètres par *grid search*.

6.5 Fidélité de la *cross-validation*
-------------------------------------

***Question 12*** : pour tous les modèles calculés, récupérer l'AUC calculé par *cross-validation* et sur l'échantillon de test. Est-ce que l'AUC "CV" est un indicateur fiable de la performance sur les données test ? Pourquoi ?

<!--
AUC       Test      CV
LogReg    0,93379
Overfit   0,92972   0,93496
Base      0,94784   0,94760
Best CV   0,94985   0,94848

          Test AUC  OOB err
RF        0,94317   0,08509
Best RF   0,94253   0,08254

6.6 Note sur le calcul d'importance des forêts aléatoires
---------------------------------------------------------

L'importance des variables bruitées calculée pour les forêts aléatoires et **XGBoost** est donnée ci-dessous :

| Variable | Importance RF | Rang RF | Importance XGBoost | Rang XGBoost |
|----------|---------------|---------|--------------------|--------------|
| var1     | 0,6           | 10      | 0,79               | 10           |
| var2     | 1,9           | 9       | 11,4               | 3            |
| var3     | 3,0           | 8       | 4,8                | 6            |
| var4     | 5,0           | 7       | 4,4                | 9            |
| var5     | 5,6           | 6       | 11,1               | 4            |
| var6     | 9,9           | 5       | 14,5               | 1            |
| var7     | 10,3          | 4       | 6,0                | 5            |
| var8     | 10,8          | 3       | 4,6                | 7            |
| var9     | 12,6          | 2       | 4,5                | 8            |
| var10    | 17,5          | 1       | 12,7               | 2            |

Les variables bruitées ont été créées de cette façon : la variable var*i*, avec *i* allant de 1 à 10, représente un tirage aléatoire de nombres compris entre 1 et *i* + 1. Le nombre de modalités de la variable var*i* est donc *i* + 1.
Manifestement, l'importance calculée par forêts aléatoires est proportionnelle au nombre de modalités, ce qui n'est pas un effet voulu. Ce phénomène est connu : une forêt sur-estime l'importance d'une variable si elle a beaucoup de modalités. [Voir cette page](http://rnowling.github.io/machine/learning/2015/08/10/random-forest-bias.html) qui cite des travaux de recherche et qui prouve la présence de ce phénomène. En revanche, **XGBoost** semble ne pas être gêné par les variables avec un "grand" nombre de modalités.

6.7 Gain de la parallélisation
------------------------------

Le code suivant permet de montrer que sur un PC personnel, utiliser l'ensemble des 4 coeurs raccourcit le temps d'exécution d'environ 2 fois (13,1 s VS 6,6).

``` r
coeurs_xgb <- data.frame(coeurs = 1:4,
                         temps = numeric(4))

for(i in seq(4)){
  cat(paste(i, 'coeur-s utilisé-s \n'))
  coeurs_xgb$temps[i] <- system.time(xgboost(data = donnee, objective = 'reg:logistic', eval_metric = 'auc', 
                                             nthread = i, verbose = 1, print_every_n = 20,
                                             nrounds = 100, eta = 0.1, max_depth = 10))[3]
}
```

Sur le serveur, qui dispose de 12 coeurs, la parallélisation apporte un gain de temps en utilisant entre 1 et 5 coeurs. Au delà, le code s'exécute de plus en plus lentement :

| Nb coeurs | Temps d'exécution (s) |
|-----------|-----------------------|
| 1         | 16,2                  |
| 2         | 8,2                   |
| 3         | 6,3                   |
| 4         | 5,4                   |
| 5         | 5,1                   |
| 6         | 5,9                   |
| 7         | 6,3                   |
| 8         | 18,1                  |

7 Comparaison avec d'autres modèles
===================================

En évaluant les performances de chaque algorithme sur l'échantillon test (via l'AUC), nous obtenons les résultats suivants :

| Modèle                    | AUC en test |
|---------------------------|-------------|
| Rég logistique            | 93,379%     |
| Forêt aléatoire (défaut)  | 94,317%     |
| Forêt aléatoire optimisée | 94,253%     |
| XGBoost *overfit*         | 92,972%     |
| XGBoost (défaut)          | 94,784%     |
| XGBoost optimisé          | 94,985%     |

![](/assets/bg.png)
