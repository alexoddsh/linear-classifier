För LINALG SF1672 

Runna först:

´´´bash
pip install .
´´´
Eller uv/poetry eller vad ni nu använder som package manager för att ladda ner deps.För att ladda ned 

Kör sedan:

´´´bash
python data.py
´´´

En gång för att hämta datan som som behövs.

Sen är det bara att köra

´´´bash
python classifier.py --centroid
´´´

För att köra programmet med centroid metoden.

´´´bash 
python classifier.py --lp
´´´

För att köra det med LP metoden.  
