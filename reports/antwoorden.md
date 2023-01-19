# Tentamen ML2022-2023

De opdracht is om de audio van 10 cijfers, uitgesproken door zowel mannen als vrouwen, te classificeren. De dataset bevat timeseries met een wisselende lengte.

In [references/documentation.html](references/documentation.html) lees je o.a. dat elke timestep 13 features heeft.
Jouw junior collega heeft een neuraal netwerk gebouwd, maar het lukt hem niet om de accuracy boven de 67% te krijgen. Aangezien jij de cursus Machine Learning bijna succesvol hebt afgerond hoopt hij dat jij een paar betere ideeen hebt.

## Vraag 1

### 1a
In `dev/scripts` vind je de file `01_model_design.py`.
Het model in deze file heeft in de eerste hidden layer 100 units, in de tweede layer 10 units, dit heeft jouw collega ergens op stack overflow gevonden en hij had gelezen dat dit een goed model zou zijn.
De dropout staat op 0.5, hij heeft in een blog gelezen dat dit de beste settings voor dropout zou zijn.

- Wat vind je van de architectuur die hij heeft uitgekozen (een Neuraal netwerk met drie Linear layers)? Wat zijn sterke en zwakke kanten van een model als dit in het algemeen? En voor dit specifieke probleem?

### <span style='background :yellow' > Antwoord: De keuze voor een relatief simpel neuraal netwerk met drie lagen ligt voor de hand bij een simpele dataset. Deze bevat een input layer, hidden layer en een output layer. Het sterke punt is dat dit model snel en simpel een baseline creert en overfitting voorkomt. Het zwakke punt is dat de accuratie hoogstwaarschijnlijk te verbeteren is, mede omdat het gekozen model niet optimaal is voor deze data. Een ander model, zoals een CNN, zou geschikter zijn voor geluidsclassificatie met een volgorderlijkheid. </span>

- Wat vind je van de keuzes die hij heeft gemaakt in de LinearConfig voor het aantal units ten opzichte van de data? En van de dropout?

### <span style='background :yellow' > Antwoord: De input 13 is gelijk aan het aantal attributen en de output 20 gelijk aan het aantal classificaties (0 t/m 9 voor man en vrouw). Indien het alleen de cijfers betreft, zonder man/vrouw classificatie dan zou dit 10 moeten zijn. H1 100 is mogelijk 10 cijfers uitgesproken door mannen X 10 cijfers uitgesproken door vrouwen (= 100). H2 10 betreft het aantal cijfers (0 t/m 9). De dropout lijkt aan de hoge kant met 0,5 (= een verlies van de helft). Al met al lijken de keuzes te verklaren, met uitzondering van de (te) hoge dropout.</span>

## 1b
Als je in de forward methode van het Linear model kijkt (in `tentamen/model.py`) dan kun je zien dat het eerste dat hij doet `x.mean(dim=1)` is. 

- Wat is het effect hiervan? Welk probleem probeert hij hier op te lossen? (maw, wat gaat er fout als hij dit niet doet?)

### <span style='background :yellow' > Antwoord: Met de code bepaald men vanuit welke dimensie de mean wordt berekend, in dit geval de tweede dimensie. De data bestaat uit rijen met cijfers, waarvoor een oplossing moet komen. </span>

- Hoe had hij dit ook kunnen oplossen?

### <span style='background :yellow' > Antwoord: Door het toepassen van een average pooling functie in het model. VERDERE UITLEG </span>

- Wat zijn voor een nadelen van de verschillende manieren om deze stap te doen?

### <span style='background :yellow' > Antwoord: Bij de downsampling van pooling geeft Max pooling de maximum (hoogste) waarde. Een bijkomend nadeel kan zijn dat dus alleen het grootste element wordt meegenomen en de rest wordt genegeerd. Een voordeel bij eenvoudige lineaire classificaties is een betere performance.   </span>

### <span style='background :yellow' > Average pooling geeft de gemiddelde waarde. Het nadeel is een lagere kwaliteit in contrast door het gemiddelde, terwijl het voordeel is dat average pooling in staat is de plaatjes beter te exthraheren (minder effect op de waardes).  </span>

### 1c
Omdat jij de cursus Machine Learning hebt gevolgd kun jij hem uitstekend uitleggen wat een betere architectuur zou zijn.

- Beschrijf de architecturen die je kunt overwegen voor een probleem als dit. Het is voldoende als je beschrijft welke layers in welke combinaties je zou kunnen gebruiken.

### <span style='background :yellow' > Antwoord: Ik zou een model gebruiken geschikt voor een timeserie, zoals een CNN model met de volgende lagen:
---
### <span style='background :yellow' > Conv2D
### <span style='background :yellow' > Relu
### <span style='background :yellow' > Maxpool
### <span style='background :yellow' > Conc2D
### <span style='background :yellow' > Relu  
### <span style='background :yellow' > Maxpool
### <span style='background :yellow' > Conv2D
### <span style='background :yellow' > Relu
### <span style='background :yellow' > Maxpool
---
### <span style='background :yellow' > Flatten
### <span style='background :yellow' > Linear
### <span style='background :yellow' > Relu
### <span style='background :yellow' > Linear
### <span style='background :yellow' > Relu
### <span style='background :yellow' > Linear</span>
---

### <span style='background :yellow' > Het eerste blok bestaat uit drie herhalingen van een convolutional laag, een relu activatie laag en een maxpool laag. Het tweede blok wordt de data UITLEG 

### <span style='background :yellow' > 1) Convolutional layer:
### <span style='background :yellow' > 2) Pooling layer: 
### <span style='background :yellow' > 3) Fully Connected layer:
### <span style='background :yellow' > 4) Dropout: 
### <span style='background :yellow' > 5) Activation functions: </span>

- Geef vervolgens een indicatie en motivatie voor het aantal units/filters/kernelsize etc voor elke laag die je gebruikt, en hoe je omgaat met overgangen (bv van 3 naar 2 dimensies). Een indicatie is bijvoorbeeld een educated guess voor een aantal units, plus een boven en ondergrens voor het aantal units. Met een motivatie laat je zien dat jouw keuze niet een random selectie is, maar dat je 1) andere problemen hebt gezien en dit probleem daartegen kunt afzetten en 2) een besef hebt van de consquenties van het kiezen van een range.

### <span style='background :yellow' > Antwoord:  
### <span style='background :yellow' > Learning rate: Op basis van de uitkomsten tussentijdse opdracht is de instelling optimizer 0,001 het beste gebleken voor de classificatie.
### <span style='background :yellow' > Filters: 128 lijkt een mooie instelling om te starten. In de les is naar voren gekomen dat voor 10 classes, de filterinstelling rond de 100 zou moeten. Om de stappen x2 tussen 8 t/m 512 te kunnen maken is voor 128 (ipv 100) gekozen.
### <span style='background :yellow' > Kernel size: 3, is de optimale keuze om symetrische lagen te krijgen. Bij kernels 2 en 4 is dit niet mogelijk.
### <span style='background :yellow' > Loss function: Cross-entropy, omdat deze passend is voor een classificatie.
### <span style='background :yellow' > Optimizer: Adam, omdat deze optimizer 
### <span style='background :yellow' > Dimensies: Door de flatten optie te gebruiken wordt de data twee dimensionaal gemaakt.</span>

- Geef aan wat jij verwacht dat de meest veelbelovende architectuur is, en waarom (opnieuw, laat zien dat je niet random getallen noemt, of keuzes maakt, maar dat jij je keuze baseert op ervaring die je hebt opgedaan met andere problemen).

### <span style='background :yellow' > Antwoord:  </span>

### 1d
Implementeer jouw veelbelovende model: 

- Maak in `model.py` een nieuw nn.Module met jouw architectuur
- Maak in `settings.py` een nieuwe config voor jouw model
- Train het model met enkele educated guesses van parameters. 
- Rapporteer je bevindingen. Ga hier niet te uitgebreid hypertunen (dat is vraag 2), maar rapporteer (met een afbeelding in `antwoorden/img` die je linkt naar jouw .md antwoord) voor bijvoorbeeld drie verschillende parametersets hoe de train/test loss curve verloopt.

### <span style='background :yellow' > Antwoord:  </span>

- reflecteer op deze eerste verkenning van je model. Wat valt op, wat vind je interessant, wat had je niet verwacht, welk inzicht neem je mee naar de hypertuning.

### <span style='background :yellow' > Antwoord:  </span>

Hieronder een voorbeeld hoe je een plaatje met caption zou kunnen invoegen.

<figure>
  <p align = "center">
    <img src="img/motivational.png" style="width:50%">
    <figcaption align="center">
      <b> Fig 1.Een motivational poster voor studenten Machine Learning (Stable Diffusion)</b>
    </figcaption>
  </p>
</figure>

## Vraag 2
Een andere collega heeft alvast een hypertuning opgezet in `dev/scripts/02_tune.py`.

### 2a
Implementeer de hypertuning voor jouw architectuur:
- zorg dat je model geschikt is voor hypertuning
- je mag je model nog wat aanpassen, als vraag 1d daar aanleiding toe geeft. Als je in 1d een ander model gebruikt dan hier, geef je model dan een andere naam zodat ik ze naast elkaar kan zien.
- Stel dat je je model aanpast, laat dan je code zien (bv 2e model met aanpassing).
- voeg jouw model in op de juiste plek in de `tune.py` file.
- maak een zoekruimte aan met behulp van pydantic (naar het voorbeeld van LinearSearchSpace), maar pas het aan voor jouw model.
- Licht je keuzes toe: wat hypertune je, en wat niet? Waarom? En in welke ranges zoek je, en waarom? Zie ook de [docs van ray over search space](https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs) en voor [rondom search algoritmes](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#bohb-tune-search-bohb-tunebohb) voor meer opties en voorbeelden.

### 2b
- Analyseer de resultaten van jouw hypertuning; visualiseer de parameters van jouw hypertuning en sla het resultaat van die visualisatie op in `reports/img`. Suggesties: `parallel_coordinates` kan handig zijn, maar een goed gekozen histogram of scatterplot met goede kleuren is in sommige situaties duidelijker! Denk aan x en y labels, een titel en units voor de assen.
- reflecteer op de hypertuning. Wat werkt wel, wat werkt niet, wat vind je verrassend, wat zijn trade-offs die je ziet in de hypertuning, wat zijn afwegingen bij het kiezen van een uiteindelijke hyperparametersetting.

Importeer de afbeeldingen in jouw antwoorden, reflecteer op je experiment, en geef een interpretatie en toelichting op wat je ziet.

### 2c
- Zorg dat jouw prijswinnende settings in een config komen te staan in `settings.py`, en train daarmee een model met een optimaal aantal epochs, daarvoor kun je `01_model_design.py` kopieren en hernoemen naar `2c_model_design.py`.

## Vraag 3
### 3a
- fork deze repository.
- Zorg voor nette code. Als je nu `make format && make lint` runt, zie je dat alles ok is. Hoewel het in sommige gevallen prima is om een ignore toe te voegen, is de bedoeling dat je zorgt dat je code zoveel als mogelijk de richtlijnen volgt van de linters.
- We werken sinds 22 november met git, en ik heb een `git crash coruse.pdf` gedeeld in les 2. Laat zien dat je in git kunt werken, door een git repo aan te maken en jouw code daarheen te pushen. Volg de vuistregel dat je 1) vaak (ruwweg elke dertig minuten aan code) commits doet 2) kleine, logische chunks van code/files samenvoegt in een commit 3) geef duidelijke beschrijvende namen voor je commit messages
- Zorg voor duidelijke illustraties; voeg labels in voor x en y as, zorg voor eenheden op de assen, een titel, en als dat niet gaat (bv omdat het uit tensorboard komt) zorg dan voor een duidelijke caption van de afbeelding waar dat wel wordt uitgelegd.
- Laat zien dat je je vragen kort en bondig kunt beantwoorden. De antwoordstrategie "ik schiet met hagel en hoop dat het goede antwoord ertussen zit" levert minder punten op dan een kort antwoord waar je de essentie weet te vangen. 
- nodig mij uit (github handle: raoulg) voor je repository. 
