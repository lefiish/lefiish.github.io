------------------------------------------------------------------------

------------------------------------------------------------------------

0 R est-il lent ? Pourquoi ?
============================

R est considéré par de nombreuses personnes comme étant "lent", ce qui est plutôt vrai. La raison principale est que R est un langage écrit par des statisticiens pour des statisticiens, il a été créé dans le but d'être clair et simple à utiliser. Par exemple, imaginons qu'on lance ce code :

``` r
i <- 2L + 1.5
```

Si on veut additionner 2 éléments, un entier et un nombre décimal, R doit : détecter le type du 1<sup>er</sup> élément, le type du 2<sup>nd</sup>, tester s'il est possible d'appliquer la fonction `+`, décider de convertir l'entier en réel pour effectuer l'opération, créer la variable i et comprendre que i va stocker une donnée de type numérique. En C, langage dans lequel est écrit une grande partie du code R, la plupart de ces opérations sont éxecutées avant le lancement du programme.

1 Les boucles `for` versus les méthodes vectorisées
===================================================

Autant que possible, les boucles `for` sont à proscrire au profit des méthodes vectorisées. Cela revient à exécuter la boucle en C et non en R, ce qui accélère grandement les traitements. De plus, lors du traitement d'une boucle en R, il se passe encore d'autres choses : création d'un environnement, attribution des éléments dans l'environnement, etc.

``` r
boucle <- function(x){
   res <- 0
   for(i in seq(length(x)))
      res <- res + x[i]
   res
}

system.time(boucle(1:4e6))
```

    ##    user  system elapsed 
    ##    1.84    0.00    1.86

``` r
system.time(sum(as.numeric(1:4e6)))
```

    ##    user  system elapsed 
    ##    0.03    0.02    0.04

La perte de temps semble considérable : la version vectorisée avec `sum()` a un temps très faible. Notons que `as.numeric()` est utilisé parce que la somme de 1 à 3 millions dépasse la capacité allouée par R aux entiers.

2 Optimisations du code
=======================

2.1 Préallocation : déclarer *a priori* la taille des objets
------------------------------------------------------------

Lorsqu'un traitement est répété, une boucle `for` peut être utilisée et il est très souvent nécessaire de stocker les résultats dans un vecteur ou une autre structure. Il est très important de définir à l'avance la taille du vecteur utilisé : rajouter un élément en augmentant la taille du vecteur est très peu efficace. Il faut donc éviter d'utiliser `vecteur <- c(vecteur, element)` ou `cbind(element1, element2)`, entre autres.

``` r
sum1 <- function(nb){
   sum1 <- numeric(nb)
   sum1[1] <- 1
   for(i in 2:nb)
      sum1[i] <- sum1[i-1]+i
   sum1
}

sum2 <- function(nb){
   sum2 <- 1
   for(i in 2:nb)
      sum2 <- c(sum2, sum2[i-1]+i)
   sum2
}

system.time(sum1(1e5))
```

    ##    user  system elapsed 
    ##    0.11    0.00    0.11

``` r
system.time(sum2(1e5))
```

    ##    user  system elapsed 
    ##   10.25    0.02   10.28

Pas loin de 100 fois plus rapide ! (Le gain peut varier selon les ordinateurs...)

2.2 Fonctions optimisées
------------------------

`colSums()` (ainsi que `rowSums()`, `colMeans()`, `rowMeans()`) pour calculer des sommes en colonne (et autres) sont optimisées et à préférer.

2.3 Autres optimisations : indexation d'un *dataframe*, calcul hors boucle, précalcul
-------------------------------------------------------------------------------------

Indexer veut dire accéder à un élément d'une matrice ou d'un *dataframe*.

Exemple de traitement sur une table de 60.000 lignes :

| bool1 | bool2 |    a|
|:------|:------|----:|
| FALSE | FALSE |    1|
| FALSE | TRUE  |    2|
| FALSE | TRUE  |    3|
| TRUE  | TRUE  |    4|
| FALSE | FALSE |    5|
| TRUE  | FALSE |    6|

Le code suivant crée une 4<sup>e</sup> colonne en rajoutant la valeur de la colonne *a* à la valeur précédente si les 2 colonnes booléennes sont vraies. Sinon, la valeur est réinitialisée à la valeur de la colonne *a*.

``` r
cumul <- function(temp){
   for(i in 1:nrow(temp)){
      temp[i, 4] <- i
      if(i > 1){
         if(temp[i,1] & temp[i,2])
            temp[i,4] <- temp[i,3] + temp[i-1,4]
         else
            temp[i,4] <- temp[i,3]
      } else
         temp[i,4] <- temp[i,3]
   }
}
```

En **stockant** le résultat dans un vecteur, le gain est non nul :

``` r
cumul2 <- function(temp){
   resultat <- integer(nrow(temp))
   resultat[1] <- 1
   
   for(i in 1:nrow(temp)){
      if(i > 1){
         if(temp[i,1] & temp[i,2])
            resultat[i] <- temp[i,3] + resultat[i-1]
         else
            resultat[i] <- temp[i,3]
      } else
         resultat[i] <- temp[i,3]
   }
   temp$V4 <- resultat
}

system.time(cumul(temp))
```

    ##    user  system elapsed 
    ##   21.31    0.00   21.36

``` r
system.time(cumul2(temp))
```

    ##    user  system elapsed 
    ##    2.03    0.00    2.03

En **précalculant** la condition hors de la boucle et en **évitant la condition** sur la 1<sup>e</sup> ligne, le gain continue :

``` r
cumul3 <- function(temp){
   resultat <- integer(nrow(temp))
   resultat[1] <- 1
   bool <- temp[,1] & temp[,2]
   
   for(i in 2:nrow(temp)){
      if(bool[i])
         resultat[i] <- temp[i,3] + resultat[i-1]
      else
         resultat[i] <- temp[i,3]
   }
   temp$V4 <- resultat
}

system.time(cumul3(temp))
```

    ##    user  system elapsed 
    ##    0.76    0.00    0.76

En **initialisant** intelligement le résultat, on évite une nouvelle indexation et plusieurs étapes. Les résultats sont donc plus rapides :

``` r
cumul4 <- function(temp){
   resultat <- temp[,3]
   bool <- temp[,1] & temp[,2]
   
   for(i in 2:nrow(temp))
      if(bool[i])
         resultat[i] <- resultat[i] + resultat[i-1]
   
   temp$V4 <- resultat
}

system.time(cumul4(temp))
```

    ##    user  system elapsed 
    ##    0.06    0.00    0.07

Le code final est plus de 300 fois plus rapide que le premier !

3 Meilleures implémentations
============================

3.1 Import de données
---------------------

L'import de données représente souvent la 1<sup>e</sup> tâche d'un projet de datascience ou d'analyses. Selon la méthode employée (le *package* plus précisément), la rapidité de l'import peut beaucoup varier. Testons l'import d'un *csv* classique de 500.000 lignes, 30aine de colonnes (46 Mo), via différentes méthodes.

### Fonctions de base

``` r
system.time(read.csv('table.csv', sep=';'))
```

    ##    user  system elapsed 
    ##    8.45    0.23   25.30

Plus de 40 secondes. Peut-on faire mieux ? (Notons que si l'import est réalisé une 2<sup>nde</sup> fois dans une même session R, le traitement est plus rapide.)

``` r
echantillon <- read.csv('table.csv', nrows = 50, sep=';')
classes <- sapply(echantillon, class)
system.time(read.csv('table.csv', sep=';', colClasses = classes, stringsAsFactors = F, nrows = 5e5, comment.char = ''))
```

    ##    user  system elapsed 
    ##    4.80    0.06    4.89

L'import est plus rapide en spécifiant à l'avance le type des colonnes lues. Il est aussi possible de gagner un peu en forcant R à ne pas convertir les champs caractère en *facteurs*. Nous avons testé le fait de spécifier à l'avance le nombre de lignes, le gain est léger. Le fait d'ignorer les lignes en commentaire n'a pas d'impact ici, probablement parce qu'il n'y a pas de telles lignes.

### *Package* `readr`

Il existe une autre manière d'importer des données, avec la fonction `read_table()`.

``` r
system.time(read_csv('table.csv'))
```

    ## Parsed with column specification:
    ## cols(
    ##   .default = col_integer(),
    ##   var6 = col_character(),
    ##   var8 = col_character(),
    ##   var14 = col_character(),
    ##   prime = col_double()
    ## )

    ## See spec(...) for full column specifications.

    ## Warning: 848 parsing failures.
    ##  row   col   expected actual
    ## 1379 var7  an integer     NR
    ## 2548 var16 an integer     NR
    ## 3013 var7  an integer     NR
    ## 3240 var16 an integer     NR
    ## 3273 var7  an integer     NR
    ## .... ..... .......... ......
    ## See problems(...) for more details.

    ##    user  system elapsed 
    ##    2.82    0.45   42.78

On gagne presque la moitié en temps de traitement en utilisant un *package* optimisé, un peu plus de 3 fois par rapport à l'import classique.

### *Package* `data.table`

``` r
system.time(fread('table.csv', sep=';'))
```

    ## 
    Read 0.0% of 500000 rows
    Read 500000 rows and 1 (of 1) columns from 0.045 GB file in 00:00:41

    ##    user  system elapsed 
    ##    0.66    0.41   40.64

Ce *package* récent a l'avantage de diminuer environ par 4 le temps de calcul par rapport à un import classique. Les deux fonctions sont écrites en C donc la différence ne peut venir de là. La 1<sup>e</sup> fonction de base (sans arguments définis) est plus lente parce qu'elle force à lire toute la donnée en mémoire, comme s'il n'y avait que des variables caractère, et essaie ensuite de convertir les colonnes en entiers ou en réels (si besoin est). Il n'y a pas une seule raison pour laquelle cette fonction est plus rapide que les autres. C'est une succession de petites astuces qui optimisent l'import. La fonction `fread()` commence par trouver le délimiteur des colonnes et en déduit le nombre de colonnes. Les 5 premières, au milieu et dernières lignes sont utilisées pour déterminer le type des colonnes. La fonction alloue donc le bon nombre de ligne et de colonnes, avec le bon type, dès le départ, sans avoir besoin de lire tout le fichier.

4 BLAS
======

5
=

pqR fastR

6
=

Rcpp
