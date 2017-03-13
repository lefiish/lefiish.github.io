1 Importation des données
=========================

Installer le package `data.table` pour charger rapidement les données dans R (les autres packages sont très souvent moins rapides). Ce *dataset* concerne des campagnes de marketing direct afin de détecter si un client d'une banque va effectuer un dépôt à terme.

``` {.r}
data <- fread('bank-additional-full.csv', sep=';', data.table = FALSE)
```

Récupérer la cible dans un vecteur à part, et changer *yes*/*no* en 1/0.

``` {.r}
y <- data$y
data$y <- NULL

y <- (y == 'yes') %>% as.numeric
```

2 Traitement des variables
==========================

La méthode **XGBoost** ne sait pas gérer les variables caractères. Il faut donc les modifier, par exemple en utilisant l'encodage *one-hot* : pour une variable initiale donnée, on crée autant de variables qu'il y a de modalités. Ces variables sont des indicatrices de chaque modalité.

``` {.r}
classes <- data[1, ] %>% sapply(class)
char <- (classes == 'character') %>% which %>% names

for(j in char){
  print(j)
  for(i in data[, j] %>% unique){
    print(i)
    data[, paste0(j, '_', i)] <- data[, j] == i
  }
}
```

    ## [1] "job"
    ## [1] "housemaid"
    ## [1] "services"
    ## [1] "admin."
    ## [1] "blue-collar"
    ## [1] "technician"
    ## [1] "retired"
    ## [1] "management"
    ## [1] "unemployed"
    ## [1] "self-employed"
    ## [1] "unknown"
    ## [1] "entrepreneur"
    ## [1] "student"
    ## [1] "marital"
    ## [1] "married"
    ## [1] "single"
    ## [1] "divorced"
    ## [1] "unknown"
    ## [1] "education"
    ## [1] "basic.4y"
    ## [1] "high.school"
    ## [1] "basic.6y"
    ## [1] "basic.9y"
    ## [1] "professional.course"
    ## [1] "unknown"
    ## [1] "university.degree"
    ## [1] "illiterate"
    ## [1] "default"
    ## [1] "no"
    ## [1] "unknown"
    ## [1] "yes"
    ## [1] "housing"
    ## [1] "no"
    ## [1] "yes"
    ## [1] "unknown"
    ## [1] "loan"
    ## [1] "no"
    ## [1] "yes"
    ## [1] "unknown"
    ## [1] "contact"
    ## [1] "telephone"
    ## [1] "cellular"
    ## [1] "month"
    ## [1] "may"
    ## [1] "jun"
    ## [1] "jul"
    ## [1] "aug"
    ## [1] "oct"
    ## [1] "nov"
    ## [1] "dec"
    ## [1] "mar"
    ## [1] "apr"
    ## [1] "sep"
    ## [1] "day_of_week"
    ## [1] "mon"
    ## [1] "tue"
    ## [1] "wed"
    ## [1] "thu"
    ## [1] "fri"
    ## [1] "poutcome"
    ## [1] "nonexistent"
    ## [1] "failure"
    ## [1] "success"
