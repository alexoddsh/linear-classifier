För LINALG SF1672 

Runna först:

```bash
pip install -e.
```
Eller uv/poetry eller vad ni nu använder som package manager för att installera packages.

För att ladda ned datan kör sedan:

```bash
python data.py
```

En gång för att hämta datan som som behövs.

Sen är det bara att köra

```bash
python -m classifier --model centroid --mode local
```

För att köra programmet med centroid metoden lokalt.

```bash 
python -m classifier --model lp --mode modal
```

För att köra det med LP metoden. Krävs att ni har modal. Kan göras lokalt men tar timmar... 
.env var USE_GPU måste vara set till 0 (local) eller 1 (modal).

Nuvarande val Acc centroid: 99.285714%
Nuvarande val Acc LP: 