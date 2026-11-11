För LINALG SF1672 

Runna först:

```bash
pip install .
```
Eller uv/poetry eller vad ni nu använder som package manager för att ladda ner deps.

För att ladda ned datan kör sedan:

```bash
python data.py
```

En gång för att hämta datan som som behövs.

Sen är det bara att köra

```bash
python classifier.py --model centroid
```

För att köra programmet med centroid metoden.

```bash 
python classifier.py --model lp
```

För att köra det med LP metoden.  

Nuvarande val Acc centroid: 99.285714%
Nuvarande val Acc LP: 