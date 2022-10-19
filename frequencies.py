#!/usr/bin/env python3

# Author : Brendan Borne

# -------------------------------------------------------------------- #
#    PROJET MODELES STOCHASTIQUES ET PROBABILISTES EN EVOLUTION        #
# -------------------------------------------------------------------- #

print('Initialisation de la simulation... ')

# IMPORTATION DES LIBRAIRIES

# On commence simplement par importer toutes les librairies qui nous serviront lors de l'exécution de ce script.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import math
from tqdm import tqdm

# On configure seaborn pour qu'ils nous donne des graphiques corrects à inclure dans un rapport
sns.set_theme(style="white")
sns.set_palette("colorblind")
sns.set_context("paper")

# INITIALISATION DE LA SIMULATION

# On initialise les conditions de la simulation.
Ngen = 200 # Nombre de générations
Nind = 1000 # Taille de la population
Nloci = 1 # Nombre de locus
Nrep = 100 # Nombre de répétitions de trajectoires d'évolution d'un type de reproduction
mutation_rate = 1/Nind # Taux de mutation
mutation_proba = 1 - math.exp(-mutation_rate) # Transformation du taux de mutation en probabilité de mutation. Ici dt = 1
clone_proba = 0.95 # Probabilité de reproduction clonale pour le scénario 3

# On définit la fonction de calcul de la divergence de Kullback-Leibler
def KulLieb(a,b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.where(a!=0,a*np.log(a/b),0))

print('Fait!')

# -------------------------------------------------------------------- #
#                               SIMULATION                             #
# -------------------------------------------------------------------- #

print('Simulation des scénarios...')

# Préparation des sorties de la simulation
# Ici, on sortira un DataFrame de données brutes contenant tous les individus à chaque répétition.
# Cette sortie n'est pas demandée ni utile pour produire les graphiques requis lors de ce projet, mais cela permet d'avoir une sortie brute en plus.
# Pour générer les graphiques, nous sortons également un DataFrame avec les proportions pour chaque fréquence génotypique.

array_desc_scen1 = [] # Sortie brute de tous les descendants pour le scénario 1
array_prop_scen1 = [] # Sortie contenant les fréquences génotypiques pour le scénario 1

array_desc_scen2 = [] # Sortie brute de tous les descendants pour le scénario 2
array_prop_scen2 = [] # Sortie contenant les fréquences génotypiques pour le scénario 2

array_desc_scen3 = [] # Sortie brute de tous les descendants pour le scénario 3
array_prop_scen3 = [] # Sortie contenant les fréquences génotypiques pour le scénario 3

# Nous allons simuler les trois scénarios au sein du même ensemble de boucles imbriquées.
# La logique générale de la simulation est la même pour les trois scénarios.
# On commence par tirer le ou les parents (selon la situation) aléatoirement dans le vecteur 'parent'. 
# On sélectionne ensuite les allèles (encore une fois, aléatoirement) transmis.
# On définit ensuite en tirant dans une loi binomiale si notre descendant mute ou pas. On applique la mutation le cas échéant.
# On finit incrémenter un compteur du nombre d'individus selon l'association d'allèles afin de calculer les proportions pour cette génération.

# Nous travaillons sur du stochastique, ici nous ferons 100 répétitions
for rep in tqdm(range(0,Nrep)): 
    # On génère les populations de parents
    # Au point de départ, pour avoir un élément de comparaison convenable, on a la même population de parents pour les trois scénarios
    parent_scen1 = np.random.randint(2, size=(Nind,Nloci*2),dtype="int8")
    parent_scen2 = parent_scen1
    parent_scen3 = parent_scen1

    # Nous faisons évoluer notre population sur 200 générations
    for gen in range(0,Ngen): 

        count00_scen1 = 0 # Compteur de génotype 0x0 pour le scénario 1
        count11_scen1 = 0 # Compteur de génotype 1x1 pour le scénario 1
        count01_scen1 = 0 # Compteur de génotype 0x1 (ou 1x0) pour le scénario 1

        count00_scen2 = 0 # Compteur de génotype 0x0 pour le scénario 2
        count11_scen2 = 0 # Compteur de génotype 1x1 pour le scénario 2
        count01_scen2 = 0 # Compteur de génotype 0x1 (ou 1x0) pour le scénario 2

        count00_scen3 = 0 # Compteur de génotype 0x0 pour le scénario 3
        count11_scen3 = 0 # Compteur de génotype 1x1 pour le scénario 3
        count01_scen3 = 0 # Compteur de génotype 0x1 (ou 1x0) pour le scénario 3

        thisGen_scen1 = [] # Stocke la génération en cours pour changer le vecteur parent pour le scénario 1
        thisGen_scen2 = [] # Stocke la génération en cours pour changer le vecteur parent pour le scénario 2
        thisGen_scen3 = [] # Stocke la génération en cours pour changer le vecteur parent pour le scénario 3

        # Nous devons générer 1000 individus par génération
        for i in range(0,Nind): 
            
            # SCENARIO 1 - POPULATION SEXUEE PANMICTIQUE

            parent1_scen1 = random.choice(parent_scen1) # On sélectionne aléatoirement le premier parent
            parent2_scen1 = random.choice(parent_scen1) # On sélectionne aléatoirement le second parent

            # On remplit le vecteur de descendants avec le génotype du descendant actuel, la génération et la répétition en cours
            descendant_scen1 = [[random.choice(parent1_scen1),random.choice(parent2_scen1)],gen,rep] 
            # Vu que notre mutation est probabiliste, on calcule si l'on a muté ou non :
            mutation_scen1 = np.random.binomial(1, mutation_proba) # Vaut 1 si on a muté, vaut 0 sinon
            # Dans le cas où nous avons bien muté, on transforme donc l'un des allèles du descendant :
            if mutation_scen1 == 1:
                allele = np.random.binomial(1,0.5)
                if descendant_scen1[0][allele]==1:
                    descendant_scen1[0][allele]=0
                else:
                    descendant_scen1[0][allele]=1

            # Gestion du compte
            if np.array_equal(descendant_scen1[0],[0,0]):
                count00_scen1+=1
            elif np.array_equal(descendant_scen1[0],[1,1]):
                count11_scen1+=1
            else:
                count01_scen1+=1

            # On ajoute le descendant à la liste de sortie brute
            array_desc_scen1.append(descendant_scen1)

            # On peuple le vecteur de la génération en cours
            thisGen_scen1.append(descendant_scen1[0])

            # SCENARIO 2 - POPULATION SEXUEE AUTOGAME
            parent1_scen2 = random.choice(parent_scen2) # On sélectionne aléatoirement le parent

            # On remplit le vecteur de descendants avec le génotype du descendant actuel, la génération et la répétition en cours
            descendant_scen2 = [[random.choice(parent1_scen2),random.choice(parent1_scen2)],gen,rep] 

            # Vu que notre mutation est probabiliste, on calcule si l'on a muté ou non :
            mutation_scen2 = np.random.binomial(1, mutation_proba) # Vaut 1 si on a muté, vaut 0 sinon
            # Dans le cas où nous avons bien muté, on transforme donc le premier loci du descendant :
            if mutation_scen2 == 1:
                allele = np.random.binomial(1,0.5)
                if descendant_scen2[0][allele]==1:
                    descendant_scen2[0][allele]=0
                else:
                    descendant_scen2[0][allele]=1

            # Gestion du compte
            if np.array_equal(descendant_scen2[0],[0,0]):
                count00_scen2+=1
            elif np.array_equal(descendant_scen2[0],[1,1]):
                count11_scen2+=1
            else:
                count01_scen2+=1

            # On ajoute le descendant à la liste de sortie brute
            array_desc_scen2.append(descendant_scen2)

            # On peuple le vecteur de la génération en cours
            thisGen_scen2.append(descendant_scen2[0])

            # SCENARIO 3 - POPULATION 95% CLONALE

            parent1_scen3 = random.choice(parent_scen3) # On sélectionne aléatoirement le premier parent
            parent2_scen3 = random.choice(parent_scen3) # On sélectionne aléatoirement le second parent

            # On vérifie si on est dans le cas où le parent sera cloné :
            clone = np.random.binomial(1, clone_proba)
            if clone == 1:
                descendant_scen3 = [parent1_scen3,gen,rep]
            else:
                # On remplit le vecteur de descendants avec le génotype du descendant actuel, la génération et la répétition en cours
                descendant_scen3 = [[random.choice(parent1_scen3),random.choice(parent2_scen3)],gen,rep]

            # Vu que notre mutation est probabiliste, on calcule si l'on a muté ou non :
            mutation_scen3 = np.random.binomial(1, mutation_proba) # Vaut 1 si on a muté, vaut 0 sinon
            # Dans le cas où nous avons bien muté, on transforme donc le premier loci du descendant :
            if mutation_scen3 == 1:
                allele = np.random.binomial(1,0.5)
                if descendant_scen3[0][allele]==1:
                    descendant_scen3[0][allele]=0
                else:
                    descendant_scen3[0][allele]=1

            # Gestion du compte
            if np.array_equal(descendant_scen3[0],[0,0]):
                count00_scen3+=1
            elif np.array_equal(descendant_scen3[0],[1,1]):
                count11_scen3+=1
            else:
                count01_scen3+=1

            # On ajoute le descendant à la liste de sortie brute
            array_desc_scen3.append(descendant_scen3)

            # On peuple le vecteur de la génération en cours
            thisGen_scen3.append(descendant_scen3[0])

        # On change le vecteur parent pour qu'il corresponde à la génération qui vient d'être générée
        parent_scen1 = thisGen_scen1 
        parent_scen2 = thisGen_scen2
        parent_scen3 = thisGen_scen3

        prop00_scen1 = count00_scen1/Nind # Proportion d'individus 0x0 à ce pas de temps pour le scénario 1
        prop11_scen1 = count11_scen1/Nind # Proportion d'individus 1x1 à ce pas de temps pour le scénario 1
        prop01_scen1 = count01_scen1/Nind # Proportion d'individus 0x1 (ou 1x0) à ce pas de temps pour le scénario 1

        prop00_scen2 = count00_scen2/Nind # Proportion d'individus 0x0 à ce pas de temps pour le scénario 2
        prop11_scen2 = count11_scen2/Nind # Proportion d'individus 1x1 à ce pas de temps pour le scénario 2
        prop01_scen2 = count01_scen2/Nind # Proportion d'individus 0x1 (ou 1x0) à ce pas de temps pour le scénario 2

        prop00_scen3 = count00_scen3/Nind # Proportion d'individus 0x0 à ce pas de temps pour le scénario 3
        prop11_scen3 = count11_scen3/Nind # Proportion d'individus 1x1 à ce pas de temps pour le scénario 3
        prop01_scen3 = count01_scen3/Nind # Proportion d'individus 0x1 (ou 1x0) à ce pas de temps pour le scénario 3

        # On peuple nos listes de sorties de fréquences avec les proportions que l'on vient de calculer
        array_prop_scen1.append([prop00_scen1,prop11_scen1,prop01_scen1,gen,rep])
        array_prop_scen2.append([prop00_scen2,prop11_scen2,prop01_scen2,gen,rep])
        array_prop_scen3.append([prop00_scen3,prop11_scen3,prop01_scen3,gen,rep])

print('Fait!')
# -------------------------------------------------------------------- #
#                               SORTIES                                #
# -------------------------------------------------------------------- #

print('Génération des sorties brutes... ')

# Mise en forme des sorties de la simulation pour chaque scénario.
# On utilise le format d'un DataFrame de la librairie pandas.
# Ce n'est pas obligatoire mais je trouve le travail plus facile sur des DataFrame pandas,
# notamment pendant les phases de recherche sur jupyter notebook, l'affichage est bien plus agréable.
# C'est également un format qui permet de générer des graphiques élégants assez simplement grâce à Seaborn.
# Enfin, cela permet d'ecrire un fichier .csv de sortie brutes.
df_descendant_scen1 = pd.DataFrame(array_desc_scen1,columns=["LL","gen","rep"])
df_props_scen1 = pd.DataFrame(array_prop_scen1,columns=["00","11","01","gen","rep"])
df_props_scen1 = df_props_scen1.melt(id_vars=['gen','rep'],var_name='LL')

df_descendant_scen2 = pd.DataFrame(array_desc_scen2,columns=["LL","gen","rep"])
df_props_scen2 = pd.DataFrame(array_prop_scen2,columns=["00","11","01","gen","rep"])
df_props_scen2 = df_props_scen2.melt(id_vars=['gen','rep'],var_name='LL')

df_descendant_scen3 = pd.DataFrame(array_desc_scen3,columns=["LL","gen","rep"])
df_props_scen3 = pd.DataFrame(array_prop_scen3,columns=["00","11","01","gen","rep"])
df_props_scen3 = df_props_scen3.melt(id_vars=['gen','rep'],var_name='LL')

# Ecriture des fichiers de sortie brutes
df_descendant_scen1.to_csv('sorties_brutes_scen1.csv')
df_descendant_scen2.to_csv('sorties_brutes_scen2.csv')
df_descendant_scen3.to_csv('sorties_brutes_scen3.csv')

print('Fait!')

# SORTIES GRAPHIQUES
print('Création et écriture des sorties graphiques... ')

# La création de nos sorties graphiques nécessitent un peu de travail.
# On doit d'abord calculer la moyenne de nos fréquences génotypiques, et extraire trois trajectoires.
# Ici, je prends les trajectoirs 0, 50 et 99. On aurait pu choisir de les tirer aléatoirement, de prendre les trois premières, ou n'importe lesquelles.
# Cela ne change à priori rien puisque notre modèle est stochastique.

# Scénario 1
# On calcule les moyennes de chaque fréquence à chaque génération
df_means_scen1 = df_props_scen1.groupby(['gen','LL'],as_index=False).mean().drop(columns=['rep'])

# On créé le DataFrame qui nous permettra de faire notre graphique
df_plot_scen1 = df_means_scen1
df_plot_scen1 = df_plot_scen1.rename(columns={'value':'mean'}) # On renomme la valeur en 'mean'

df_traj1_scen1 = df_props_scen1.query('rep==0') # On récupère notre première trajectoire
df_traj2_scen1 = df_props_scen1.query('rep==50') # On récupère notre deuxième trajectoire
df_traj3_scen1 = df_props_scen1.query('rep==99') # On récupère notre troisième trajectoire

# On merge toutes nos trajectoires (moyenne et 1,2,3)
df_plot_scen1 = df_plot_scen1.merge(df_traj1_scen1) 
df_plot_scen1 = df_plot_scen1.rename(columns={'value':'traj1'}) # On renomme la valeur en 'traj1'
df_plot_scen1 = df_plot_scen1.drop(columns={'rep'}) # Nous n'aurons pas besoin de rep

df_plot_scen1 = df_plot_scen1.merge(df_traj2_scen1)
df_plot_scen1 = df_plot_scen1.rename(columns={'value':'traj2'}) # On renomme la valeur en 'traj2'
df_plot_scen1 = df_plot_scen1.drop(columns={'rep'}) # Nous n'aurons pas besoin de rep

df_plot_scen1 = df_plot_scen1.merge(df_traj3_scen1)
df_plot_scen1 = df_plot_scen1.rename(columns={'value':'traj3'}) # On renomme la valeur en 'traj3'
df_plot_scen1 = df_plot_scen1.drop(columns={'rep'}) # Nous n'aurons pas besoin de rep

# On transforme les données pour qu'elles soient au format bien compris par seaborn
df_plot_scen1 = df_plot_scen1.melt(id_vars=['gen','LL'],var_name='plot')

# On créé la figure vide
plt.figure() 
# On fait le plot
plot = sns.lineplot(x="gen",y="value",hue='LL',style='plot',linewidth=0.5,data=df_plot_scen1) 
# On le rend un peu plus joli
sns.despine(trim=True) 
plot.set_xlabel("Génération")
plot.set_ylabel("Proportion")
plot.set_title("Population sexuée panmictique")
# On exporte le pdf
plt.savefig('scenario1.pdf')

# Scénario 2
# On calcule les moyennes de chaque fréquence à chaque génération
df_means_scen2 = df_props_scen2.groupby(['gen','LL'],as_index=False).mean().drop(columns=['rep'])

# On créé le DataFrame qui nous permettra de faire notre graphique
df_plot_scen2 = df_means_scen2
df_plot_scen2 = df_plot_scen2.rename(columns={'value':'mean'})

df_traj1_scen2 = df_props_scen2.query('rep==0') # On récupère notre première trajectoire
df_traj2_scen2 = df_props_scen2.query('rep==50') # On récupère notre deuxième trajectoire
df_traj3_scen2 = df_props_scen2.query('rep==99') # On récupère notre troisième trajectoire

# On merge toutes nos trajectoires (moyenne et 1,2,3)
df_plot_scen2 = df_plot_scen2.merge(df_traj1_scen2)
df_plot_scen2 = df_plot_scen2.rename(columns={'value':'traj1'}) # On renomme la valeur en 'traj1'
df_plot_scen2 = df_plot_scen2.drop(columns={'rep'}) # Nous n'aurons pas besoin de rep

df_plot_scen2 = df_plot_scen2.merge(df_traj2_scen2)
df_plot_scen2 = df_plot_scen2.rename(columns={'value':'traj2'}) # On renomme la valeur en 'traj2'
df_plot_scen2 = df_plot_scen2.drop(columns={'rep'}) # Nous n'aurons pas besoin de rep

df_plot_scen2 = df_plot_scen2.merge(df_traj3_scen2)
df_plot_scen2 = df_plot_scen2.rename(columns={'value':'traj3'}) # On renomme la valeur en 'traj3'
df_plot_scen2 = df_plot_scen2.drop(columns={'rep'}) # Nous n'aurons pas besoin de rep

# On transforme les données pour qu'elles soient au format bien compris par seaborn
df_plot_scen2 = df_plot_scen2.melt(id_vars=['gen','LL'],var_name='plot')

# On créé la figure vide
plt.figure()
# On fait le plot
plot = sns.lineplot(x="gen",y="value",hue='LL',style='plot',linewidth=0.5,data=df_plot_scen2)
# On le rend plus joli
sns.despine(trim=True)
plot.set_xlabel("Génération")
plot.set_ylabel("Proportion")
plot.set_title('Population sexuée autogame')
# On exporte le pdf
plt.savefig('scenario2.pdf')

# Scénario 3
# On calcule les moyennes de chaque fréquence à chaque génération
df_means_scen3 = df_props_scen3.groupby(['gen','LL'],as_index=False).mean().drop(columns=['rep'])

# On créé le DataFrame qui nous permettra de faire notre graphique
df_plot_scen3 = df_means_scen3
df_plot_scen3 = df_plot_scen3.rename(columns={'value':'mean'})

df_traj1_scen3 = df_props_scen3.query('rep==0') # On récupère notre première trajectoire
df_traj2_scen3 = df_props_scen3.query('rep==50') # On récupère notre deuxième trajectoire
df_traj3_scen3 = df_props_scen3.query('rep==99') # On récupère notre troisième trajectoire

# On merge toutes nos trajectoires (moyenne et 1,2,3)
df_plot_scen3 = df_plot_scen3.merge(df_traj1_scen3)
df_plot_scen3 = df_plot_scen3.rename(columns={'value':'traj1'}) # On renomme la valeur en 'traj1'
df_plot_scen3 = df_plot_scen3.drop(columns={'rep'}) # Nous n'aurons pas besoin de rep

df_plot_scen3 = df_plot_scen3.merge(df_traj2_scen3)
df_plot_scen3 = df_plot_scen3.rename(columns={'value':'traj2'}) # On renomme la valeur en 'traj2'
df_plot_scen3 = df_plot_scen3.drop(columns={'rep'}) # Nous n'aurons pas besoin de rep

df_plot_scen3 = df_plot_scen3.merge(df_traj3_scen3)
df_plot_scen3 = df_plot_scen3.rename(columns={'value':'traj3'}) # On renomme la valeur en 'traj3'
df_plot_scen3 = df_plot_scen3.drop(columns={'rep'}) # Nous n'aurons pas besoin de rep

# On transforme les données pour qu'elles soient au format bien compris par seaborn
df_plot_scen3 = df_plot_scen3.melt(id_vars=['gen','LL'],var_name='plot')

# On créé la figure vide
plt.figure()
# On fait le plot
plot = sns.lineplot(x="gen",y="value",hue='LL',style='plot',linewidth=0.5,data=df_plot_scen3)
# On le rend plus joli
sns.despine(trim=True)
plot.set_xlabel("Génération")
plot.set_ylabel("Proportion")
plot.set_title("Population 95% clonale")
# On exporte le pdf
plt.savefig('scenario3.pdf')

# Visualisations 'bonus' avec intervalle de confiance à 95%
# Ces visualisations sont un peu plus lisibles qu'en affichant plusieurs trajectoires.
# Scénario 1
plt.figure()
plot = sns.lineplot(x='gen',y='value',hue='LL',linewidth=0.5,data=df_props_scen1)
sns.despine(trim=True)
plot.set_xlabel("Génération")
plot.set_ylabel("Proportion")
plot.set_title("Population sexuée panmictique")
plt.savefig('scenario1_ci.pdf')

# Scénario 2
plt.figure()
plot = sns.lineplot(x='gen',y='value',hue='LL',linewidth=0.5,data=df_props_scen2)
sns.despine(trim=True)
plot.set_xlabel("Génération")
plot.set_ylabel("Proportion")
plot.set_title("Population sexuée autogame")
plt.savefig('scenario2_ci.pdf')

# Scénario 3
plt.figure()
plot = sns.lineplot(x='gen',y='value',hue='LL',linewidth=0.5,data=df_props_scen3)
sns.despine(trim=True)
plot.set_xlabel("Génération")
plot.set_ylabel("Proportion")
plot.set_title("Population sexuée panmictique")
plt.savefig('scenario3_ci.pdf')

print('Fait!')

# -------------------------------------------------------------------- #
#                           KULLBACK-LEIBLER                           #
# -------------------------------------------------------------------- #
print('Calcul des divergences de Kullback-Leibler...')

# A 5 générations
# On récupère les fréquences à la génération 5 pour chaque génotype
het0x1_scen1 = df_means_scen1.query('LL=="01" and gen==4')['value'].values
het0x1_scen2 = df_means_scen2.query('LL=="01" and gen==4')['value'].values
het0x1_scen3 = df_means_scen3.query('LL=="01" and gen==4')['value'].values

hom0x0_scen1 = df_means_scen1.query('LL=="00" and gen==4')['value'].values
hom0x0_scen2 = df_means_scen2.query('LL=="00" and gen==4')['value'].values
hom0x0_scen3 = df_means_scen3.query('LL=="00" and gen==4')['value'].values

hom1x1_scen1 = df_means_scen1.query('LL=="11" and gen==4')['value'].values
hom1x1_scen2 = df_means_scen2.query('LL=="11" and gen==4')['value'].values
hom1x1_scen3 = df_means_scen3.query('LL=="11" and gen==4')['value'].values

# Puis on calcule les divergences de Kullback-Leibler
div_12 = KulLieb([het0x1_scen1,hom0x0_scen1,hom1x1_scen1],[het0x1_scen2,hom0x0_scen2,hom1x1_scen2])
div_13 = KulLieb([het0x1_scen1,hom0x0_scen1,hom1x1_scen1],[het0x1_scen3,hom0x0_scen3,hom1x1_scen3])
div_23 = KulLieb([het0x1_scen2,hom0x0_scen2,hom1x1_scen2],[het0x1_scen3,hom0x0_scen3,hom1x1_scen3])

# On les met dans un DataFrame que l'on exporte
df_KB = pd.DataFrame([[div_12,div_13,div_23]], columns=["Scen1-Scen2","Scen1-Scen3","Scen2-Scen3"])
df_KB.to_csv('KB_5gen.csv')

# Et qu'on affiche ensuite tout de même dans le terminal
print('Divergences de Kullback-Leibler à 5 générations:')
print(df_KB)

# A 20 générations
# On récupère les fréquences à la génération 20 pour chaque génotype
het0x1_scen1 = df_means_scen1.query('LL=="01" and gen==19')['value'].values
het0x1_scen2 = df_means_scen2.query('LL=="01" and gen==19')['value'].values
het0x1_scen3 = df_means_scen3.query('LL=="01" and gen==19')['value'].values

hom0x0_scen1 = df_means_scen1.query('LL=="00" and gen==19')['value'].values
hom0x0_scen2 = df_means_scen2.query('LL=="00" and gen==19')['value'].values
hom0x0_scen3 = df_means_scen3.query('LL=="00" and gen==19')['value'].values

hom1x1_scen1 = df_means_scen1.query('LL=="11" and gen==19')['value'].values
hom1x1_scen2 = df_means_scen2.query('LL=="11" and gen==19')['value'].values
hom1x1_scen3 = df_means_scen3.query('LL=="11" and gen==19')['value'].values

# Puis on calcule les divergences de Kullback-Leibler
div_12 = KulLieb([het0x1_scen1,hom0x0_scen1,hom1x1_scen1],[het0x1_scen2,hom0x0_scen2,hom1x1_scen2])
div_13 = KulLieb([het0x1_scen1,hom0x0_scen1,hom1x1_scen1],[het0x1_scen3,hom0x0_scen3,hom1x1_scen3])
div_23 = KulLieb([het0x1_scen2,hom0x0_scen2,hom1x1_scen2],[het0x1_scen3,hom0x0_scen3,hom1x1_scen3])

# On les met dans un DataFrame que l'on exporte
df_KB = pd.DataFrame([[div_12,div_13,div_23]], columns=["Scen1-Scen2","Scen1-Scen3","Scen2-Scen3"])
df_KB.to_csv('KB_20gen.csv')

# Et qu'on affiche ensuite tout de même dans le terminal
print('Divergences de Kullback-Leibler à 20 générations:')
print(df_KB)

print('Fin du programme.')