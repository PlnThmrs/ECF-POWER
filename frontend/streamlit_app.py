import requests
import streamlit as st

st.title("Prédiction de la consommation d'énergie des maisons")
st.write("Saisissez les caractéristiques du quartier pour obtenir une estimation.")

# Organisation en colonnes pour une meilleure interface
col1, col2 = st.columns(2)

with col1:
    moyenne_conso = st.number_input(
        "Consommation moyenne d'énergie(en KWH)", value=3.5, step=0.1
    )
    age_bat = st.number_input("Âge médian du bâtiment", value=20.0, step=1.0)
    surf_bat = st.number_input("Superficie bâtiment", value=500.0, step=0.1)
    nmb_pieces = st.number_input("Nombre de pièces", value=1.0, step=0.1)

with col2:
    type_chauffage = st.selectbox(
        "Type de chauffage utilisé",
        [
            "Électricité",
            "Gaz naturel",
            "Fioul",
            "Réseau de Chauffage Urbain",
            "Bois",
            "Pompe à chaleur",
            "GPL",
            "Propane",
        ],
    )

    nb_occupants = st.number_input("Nombre d'occupants du logement", value=3, step=1)

    zone_climatique = st.selectbox("Zone climatique", ["H1", "H2", "H3"])

    etiquette_dpe = st.selectbox("Étiquette DPE", ["A", "B", "C", "D", "E", "F", "G"])


if st.button("Calculer l'estimation", type="primary"):

    payload = {
        "MoyenneConso": moyenne_conso,
        "AgeBat": age_bat,
        "SurfBat": surf_bat,
        "NmbPieces": nmb_pieces,
        "TypeChauffage": type_chauffage,
        "NbOccupants": nb_occupants,
        "ZoneClimatique": zone_climatique,
        "EtiquetteDPE": etiquette_dpe,
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict", json=payload, timeout=5
        )
        response.raise_for_status()
        prediction = response.json()["prediction"]

        st.success(f"### Consommation estimée : {prediction:.2f} 100k")
        st.metric("Estimation", f"{prediction * 100_000:,.0f} $")

    except requests.exceptions.ConnectionError:
        st.error("Erreur : Impossible de contacter le serveur backend (FastAPI).")
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
