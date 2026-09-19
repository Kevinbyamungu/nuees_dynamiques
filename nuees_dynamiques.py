
import csv
import numpy as np


class NuesDynamiques:
    """
    Implementation pedagogique des Nuees Dynamiques.

    Representations disponibles :
        - "point"        : un centroide par classe -> k-means
        - "points"       : plusieurs points representatifs (noyau)
        - "axes"         : sous-espace affine defini par des axes factoriels
        - "distribution" : loi gaussienne estimee pour chaque classe
        - "structure"    : structure ellipsoidale geometrique
    """

    REPRESENTATIONS = {"point", "points", "axes", "distribution", "structure"}

    def __init__(
        self,
        n_clusters,
        representation="points",
        n_representants=3,
        n_axes=1,
        max_iter=100,
        tol=1e-6,
        regularisation=1e-6,
        random_state=0,
        verbose=True,
    ):
        if n_clusters < 2:
            raise ValueError("n_clusters doit etre >= 2.")
        if representation not in self.REPRESENTATIONS:
            raise ValueError(
                "representation doit etre parmi : "
                + ", ".join(sorted(self.REPRESENTATIONS))
            )
        if n_representants < 1:
            raise ValueError("n_representants doit etre >= 1.")
        if n_axes < 1:
            raise ValueError("n_axes doit etre >= 1.")

        self.n_clusters = int(n_clusters)
        self.representation = representation
        self.n_representants = int(n_representants)
        self.n_axes = int(n_axes)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.regularisation = float(regularisation)
        self.random_state = random_state
        self.verbose = verbose

        self.labels_ = None
        self.representations_ = None
        self.objective_ = None
        self.n_iter_ = None
        self.history_ = []

    @staticmethod
    def _distance_euclidienne_carre(X, centres):
        diff = X[:, None, :] - centres[None, :, :]
        return np.sum(diff * diff, axis=2)

    def _initialiser_labels(self, X, rng):
        n = X.shape[0]
        if self.n_clusters > n:
            raise ValueError(
                "Le nombre de classes ne peut pas depasser le nombre d'individus."
            )

        indices = rng.choice(n, size=self.n_clusters, replace=False)
        graines = X[indices]
        distances = self._distance_euclidienne_carre(X, graines)
        labels = np.argmin(distances, axis=1)

        # Garantit qu'aucune classe n'est vide au depart.
        labels[indices] = np.arange(self.n_clusters)
        return labels

    def _reparer_classes_vides(self, X, labels, distances):
        labels = labels.copy()
        effectifs = np.bincount(labels, minlength=self.n_clusters)

        while np.any(effectifs == 0):
            classe_vide = int(np.where(effectifs == 0)[0][0])

            # On prend un point mal represente dans une classe ayant au moins 2 points.
            couts = distances[np.arange(X.shape[0]), labels].copy()
            ordre = np.argsort(couts)[::-1]

            deplace = False
            for idx in ordre:
                ancienne = int(labels[idx])
                if effectifs[ancienne] > 1:
                    labels[idx] = classe_vide
                    effectifs[ancienne] -= 1
                    effectifs[classe_vide] += 1
                    deplace = True
                    break

            if not deplace:
                raise RuntimeError("Impossible de reparer une classe vide.")

        return labels

    def _representation_point(self, Xc):
        return {"centre": np.mean(Xc, axis=0)}

    def _representation_points(self, Xc):
        """
        Construit un noyau de plusieurs points representatifs.

        La distance d'un individu a une classe sera la distance au point
        representatif le plus proche. Les representants sont choisis de
        maniere gloutonne pour reduire la somme de ces distances.
        """
        m = min(self.n_representants, Xc.shape[0])

        diff = Xc[:, None, :] - Xc[None, :, :]
        D = np.sum(diff * diff, axis=2)

        premier = int(np.argmin(np.sum(D, axis=1)))
        selection = [premier]
        distance_plus_proche = D[:, premier].copy()

        while len(selection) < m:
            meilleur = None
            meilleur_cout = np.inf

            for candidat in range(Xc.shape[0]):
                if candidat in selection:
                    continue
                cout = np.sum(np.minimum(distance_plus_proche, D[:, candidat]))
                if cout < meilleur_cout:
                    meilleur_cout = cout
                    meilleur = candidat

            selection.append(meilleur)
            distance_plus_proche = np.minimum(
                distance_plus_proche, D[:, meilleur]
            )

        return {
            "representants": Xc[np.array(selection)].copy(),
            "indices_locaux": np.array(selection, dtype=int),
        }

    def _representation_axes(self, Xc):
        centre = np.mean(Xc, axis=0)
        Z = Xc - centre
        d = Xc.shape[1]

        # Un sous-espace de dimension d donnerait une distance residuelle nulle.
        q_max = max(0, min(d - 1, Xc.shape[0] - 1))
        q = min(self.n_axes, q_max)

        if q == 0:
            axes = np.empty((d, 0))
            valeurs = np.empty(0)
        else:
            covariance = (Z.T @ Z) / max(Xc.shape[0], 1)
            valeurs, vecteurs = np.linalg.eigh(covariance)
            ordre = np.argsort(valeurs)[::-1][:q]
            valeurs = valeurs[ordre]
            axes = vecteurs[:, ordre]

        return {
            "centre": centre,
            "axes": axes,
            "valeurs_propres": valeurs,
        }

    def _covariance_regularisee(self, Xc):
        centre = np.mean(Xc, axis=0)
        Z = Xc - centre
        d = Xc.shape[1]

        if Xc.shape[0] <= 1:
            covariance = np.eye(d)
        else:
            covariance = (Z.T @ Z) / Xc.shape[0]

        # Regularisation proportionnelle a l'echelle des variables.
        trace_moyenne = np.trace(covariance) / d if d > 0 else 1.0
        echelle = max(float(trace_moyenne), 1.0)
        covariance = covariance + self.regularisation * echelle * np.eye(d)
        return centre, covariance

    def _representation_distribution(self, Xc):
        centre, covariance = self._covariance_regularisee(Xc)
        signe, logdet = np.linalg.slogdet(covariance)

        if signe <= 0:
            covariance = covariance + 10.0 * self.regularisation * np.eye(
                covariance.shape[0]
            )
            signe, logdet = np.linalg.slogdet(covariance)

        inverse = np.linalg.pinv(covariance)

        return {
            "moyenne": centre,
            "covariance": covariance,
            "inverse": inverse,
            "logdet": float(logdet),
        }

    def _representation_structure(self, Xc):
        """
        Structure representative concrete : ellipsoide geometrique.

        On estime la forme de la classe par une matrice de covariance,
        puis on la normalise pour separer la notion de forme de celle
        d'une densite probabiliste.
        """
        centre, covariance = self._covariance_regularisee(Xc)
        d = covariance.shape[0]

        signe, logdet = np.linalg.slogdet(covariance)
        if signe <= 0:
            covariance = covariance + 10.0 * self.regularisation * np.eye(d)
            signe, logdet = np.linalg.slogdet(covariance)

        # Normalisation du volume : det(forme) = 1.
        facteur = np.exp(logdet / d)
        forme = covariance / facteur
        inverse_forme = np.linalg.pinv(forme)

        return {
            "centre": centre,
            "forme": forme,
            "inverse_forme": inverse_forme,
        }

    def _construire_representation(self, Xc):
        if self.representation == "point":
            return self._representation_point(Xc)
        if self.representation == "points":
            return self._representation_points(Xc)
        if self.representation == "axes":
            return self._representation_axes(Xc)
        if self.representation == "distribution":
            return self._representation_distribution(Xc)
        if self.representation == "structure":
            return self._representation_structure(Xc)
        raise RuntimeError("Representation inconnue.")

    def _mettre_a_jour_representations(self, X, labels):
        reps = []
        for k in range(self.n_clusters):
            Xc = X[labels == k]
            if Xc.shape[0] == 0:
                raise RuntimeError("Classe vide pendant la mise a jour.")
            reps.append(self._construire_representation(Xc))
        return reps

    def _distance_point(self, X, rep):
        diff = X - rep["centre"]
        return np.sum(diff * diff, axis=1)

    def _distance_points(self, X, rep):
        R = rep["representants"]
        diff = X[:, None, :] - R[None, :, :]
        D = np.sum(diff * diff, axis=2)
        return np.min(D, axis=1)

    def _distance_axes(self, X, rep):
        Z = X - rep["centre"]
        norme2 = np.sum(Z * Z, axis=1)
        axes = rep["axes"]

        if axes.shape[1] == 0:
            return norme2

        projections = Z @ axes
        explique = np.sum(projections * projections, axis=1)
        residu = np.maximum(norme2 - explique, 0.0)
        return residu

    def _distance_distribution(self, X, rep):
        Z = X - rep["moyenne"]
        mahal = np.einsum("ij,jk,ik->i", Z, rep["inverse"], Z)
        d = X.shape[1]

        # -log vraisemblance gaussienne, a une constante pres.
        return 0.5 * (
            mahal
            + rep["logdet"]
            + d * np.log(2.0 * np.pi)
        )

    def _distance_structure(self, X, rep):
        Z = X - rep["centre"]
        return np.einsum("ij,jk,ik->i", Z, rep["inverse_forme"], Z)

    def _matrice_distances(self, X, reps):
        D = np.empty((X.shape[0], self.n_clusters), dtype=float)

        for k, rep in enumerate(reps):
            if self.representation == "point":
                D[:, k] = self._distance_point(X, rep)
            elif self.representation == "points":
                D[:, k] = self._distance_points(X, rep)
            elif self.representation == "axes":
                D[:, k] = self._distance_axes(X, rep)
            elif self.representation == "distribution":
                D[:, k] = self._distance_distribution(X, rep)
            elif self.representation == "structure":
                D[:, k] = self._distance_structure(X, rep)

        return D

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X doit etre une matrice 2D.")
        if X.shape[0] < self.n_clusters:
            raise ValueError("Pas assez d'individus pour ce nombre de classes.")
        if X.shape[1] < 1:
            raise ValueError("X doit contenir au moins une variable.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contient NaN ou infini.")

        rng = np.random.default_rng(self.random_state)
        labels = self._initialiser_labels(X, rng)

        self.history_ = []
        ancien_objectif = np.inf

        for iteration in range(1, self.max_iter + 1):
            reps = self._mettre_a_jour_representations(X, labels)
            distances = self._matrice_distances(X, reps)

            nouveaux_labels = np.argmin(distances, axis=1)
            nouveaux_labels = self._reparer_classes_vides(
                X, nouveaux_labels, distances
            )

            objectif = float(
                np.sum(distances[np.arange(X.shape[0]), nouveaux_labels])
            )
            self.history_.append(objectif)

            if self.verbose:
                print(
                    f"Iteration {iteration:3d} | "
                    f"critere = {objectif:.8f}"
                )

            stable = np.array_equal(labels, nouveaux_labels)
            amelioration_faible = (
                np.isfinite(ancien_objectif)
                and abs(ancien_objectif - objectif)
                <= self.tol * (1.0 + abs(ancien_objectif))
            )

            labels = nouveaux_labels

            if stable or amelioration_faible:
                break

            ancien_objectif = objectif

        # Recalcule les representations avec la partition finale.
        reps = self._mettre_a_jour_representations(X, labels)
        distances = self._matrice_distances(X, reps)
        objectif_final = float(
            np.sum(distances[np.arange(X.shape[0]), labels])
        )

        self.labels_ = labels
        self.representations_ = reps
        self.objective_ = objectif_final
        self.n_iter_ = iteration

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        if self.representations_ is None:
            raise RuntimeError("Le modele doit d'abord etre ajuste avec fit().")

        X = np.asarray(X, dtype=float)
        distances = self._matrice_distances(X, self.representations_)
        return np.argmin(distances, axis=1)


def lire_csv_numerique(chemin, delimiteur=","):
    """
    Lit un CSV avec en-tete, detecte automatiquement les colonnes entierement
    numeriques et renvoie :
        entetes, matrice_numerique, indices_colonnes_numeriques
    """
    with open(chemin, "r", encoding="utf-8-sig", newline="") as f:
        lecteur = csv.reader(f, delimiter=delimiteur)
        lignes = list(lecteur)

    if len(lignes) < 2:
        raise ValueError("Le CSV doit contenir un en-tete et au moins une ligne.")

    entetes = [h.strip() for h in lignes[0]]
    donnees = lignes[1:]

    nb_colonnes = len(entetes)
    for ligne in donnees:
        if len(ligne) != nb_colonnes:
            raise ValueError("Le CSV contient des lignes de longueurs differentes.")

    indices_numeriques = []

    for j in range(nb_colonnes):
        ok = True
        for ligne in donnees:
            valeur = ligne[j].strip()
            if valeur == "":
                ok = False
                break
            try:
                float(valeur.replace(",", ".")) if delimiteur != "," else float(valeur)
            except ValueError:
                ok = False
                break

        if ok:
            indices_numeriques.append(j)

    if not indices_numeriques:
        raise ValueError("Aucune colonne entierement numerique detectee.")

    X = np.empty((len(donnees), len(indices_numeriques)), dtype=float)

    for i, ligne in enumerate(donnees):
        for jj, j in enumerate(indices_numeriques):
            valeur = ligne[j].strip()
            if delimiteur != ",":
                valeur = valeur.replace(",", ".")
            X[i, jj] = float(valeur)

    return entetes, X, indices_numeriques


def standardiser(X):
    moyenne = np.mean(X, axis=0)
    ecart_type = np.std(X, axis=0)

    ecart_type[ecart_type == 0.0] = 1.0
    Z = (X - moyenne) / ecart_type

    return Z, moyenne, ecart_type


def afficher_representations(modele, noms_variables):
    print("\nRepresentations finales")

    for k, rep in enumerate(modele.representations_):
        print(f"\nClasse {k}")

        if modele.representation == "point":
            print("Centroide :", rep["centre"])

        elif modele.representation == "points":
            print("Points representatifs :")
            print(rep["representants"])

        elif modele.representation == "axes":
            print("Centre :", rep["centre"])
            print("Axes factoriels :")
            print(rep["axes"])
            print("Valeurs propres :", rep["valeurs_propres"])

        elif modele.representation == "distribution":
            print("Moyenne :", rep["moyenne"])
            print("Covariance :")
            print(rep["covariance"])

        elif modele.representation == "structure":
            print("Centre :", rep["centre"])
            print("Matrice de forme ellipsoidale :")
            print(rep["forme"])

    print("\nVariables utilisees :", ", ".join(noms_variables))


def sauvegarder_resultats(chemin_sortie, labels):
    with open(chemin_sortie, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["indice", "classe"])
        for i, classe in enumerate(labels):
            writer.writerow([i, int(classe)])


def choisir_representation():
    choix = {
        "1": "point",
        "2": "points",
        "3": "axes",
        "4": "distribution",
        "5": "structure",
    }

    print("\nType de representation de la nuee")
    print("1 - Un point : k-means")
    print("2 - Plusieurs points representatifs")
    print("3 - Axes factoriels")
    print("4 - Distribution de probabilite gaussienne")
    print("5 - Structure representative ellipsoidale")

    valeur = input("Votre choix : ").strip()

    if valeur not in choix:
        raise ValueError("Choix invalide.")

    return choix[valeur]


def main():
    print("ALGORITHME DES NUEES DYNAMIQUES")
    print("--------------------------------")

    chemin = input("Chemin du fichier CSV : ").strip().strip('"')
    delimiteur = input(
        "Delimiteur du CSV [Entree pour virgule, ; pour point-virgule] : "
    ).strip()
    if delimiteur == "":
        delimiteur = ","

    entetes, X_auto, indices_auto = lire_csv_numerique(chemin, delimiteur)

    print("\nColonnes numeriques detectees :")
    for position, indice_original in enumerate(indices_auto):
        print(f"{position} - {entetes[indice_original]}")

    saisie = input(
        "Indices des colonnes a utiliser, separes par des virgules "
        "[Entree = toutes] : "
    ).strip()

    if saisie == "":
        positions = list(range(len(indices_auto)))
    else:
        positions = [int(x.strip()) for x in saisie.split(",")]

    if any(p < 0 or p >= len(indices_auto) for p in positions):
        raise ValueError("Indice de colonne invalide.")

    X = X_auto[:, positions]
    noms_variables = [entetes[indices_auto[p]] for p in positions]

    rep = choisir_representation()
    k = int(input("Nombre de classes : ").strip())

    n_representants = 3
    n_axes = 1

    if rep == "points":
        n_representants = int(
            input("Nombre de points representatifs par classe : ").strip()
        )

    if rep == "axes":
        n_axes = int(input("Nombre d'axes factoriels par classe : ").strip())

    reponse = input("Standardiser les variables ? [o/n, defaut=o] : ").strip().lower()
    utiliser_standardisation = reponse != "n"

    if utiliser_standardisation:
        X_modele, moyenne, ecart_type = standardiser(X)
    else:
        X_modele = X.copy()
        moyenne = None
        ecart_type = None

    modele = NuesDynamiques(
        n_clusters=k,
        representation=rep,
        n_representants=n_representants,
        n_axes=n_axes,
        max_iter=100,
        tol=1e-7,
        regularisation=1e-6,
        random_state=42,
        verbose=True,
    )

    labels = modele.fit_predict(X_modele)

    print("\nResultat final")
    print("Nombre d'iterations :", modele.n_iter_)
    print("Critere final :", modele.objective_)

    effectifs = np.bincount(labels, minlength=k)
    for classe, effectif in enumerate(effectifs):
        print(f"Classe {classe} : {effectif} individus")

    afficher_representations(modele, noms_variables)

    sortie = "resultats_nuees_dynamiques.csv"
    sauvegarder_resultats(sortie, labels)
    print(f"\nAffectations sauvegardees dans : {sortie}")


if __name__ == "__main__":
    main()
