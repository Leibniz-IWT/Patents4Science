# Copyright (c) 2025 Leibniz-IWT, Norbert Riefler <riefler@iwt.uni-bremen.de>,
#                                 
# MIT Licence:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Use "AM-OntologyLiterature_spaCy"-environment for conda


################################################################################################
#from pyalex import Works, Authors, Institutions, Concepts
from pandas_ods_reader import read_ods

import pyalex

import pymupdf
import pdf2doi

import glob
import os

import spacy
from spacy.symbols import ORTH

#JSON output:
import json
#pretty output files:
import pprint
################################################################################################





################################################################################################
#File details:

#directory with pdfs:
directory = '<working directory with pdfs>'

# Begriffe in Spreadsheet mit AM-Ontologie:
table_input_dir = 'AM_EntityTypes-2025-06-26+PMDco3.0.0-FullEntitiesList-V1.ods'

# AM Ontology Entity_Types Definitionen:
definitions_input_dir = 'AM_EntityTypes-2025-08-01+PMDco3.0.0-V1.ods'

#output file with resulting entities+entity_types:
fout="Test-Result.json"
################################################################################################





################################################################################################
#Preprocessing:

#Suppress output of pdf2doi:
pdf2doi.config.set('verbose',False)

#load LLM:
nlp = spacy.load("en_core_web_sm")  # Sprachmodell von Spacy
nlp.add_pipe("sentencizer", before="parser")
ruler = nlp.add_pipe("entity_ruler", before="ner")
nlp.remove_pipe("ner")
################################################################################################





################################################################################################
# Klassen aus Tabelle ziehen
#df = read_ods(table_input_dir, 1, columns=['entity','entity_type','Thing','SubClass1','SubClass2'])
df = read_ods(table_input_dir, 1, columns=['entity','entity_type','Thing','SubClass1','SubClass2','SubClass3','SubClass4','SubClass5','SubClass6','SubClass7','SubClass8','SubClass9'])
################################################################################################





################################################################################################
def get_open_alex_ids(dois):
    """
    From a list of Digital Object Identifiers obtain their respective OpenAlex IDs.

    Args:
        dois(list): List of DOIs of publications.

    Returns:
        set: The OpenAlex IDs of the publications.

    """
    return {pyalex.Works()[doi]["id"] for doi in dois}
################################################################################################





################################################################################################
# Aus AMOntology+ProtegeInput
def to_camel_case(text):
    s = text.replace("-", " ").replace("_", " ").replace("'", "")
    s = s.split()
    if len(text) == 0:
        return text
    return s[0] + ''.join(i.capitalize() for i in s[1:])
################################################################################################





################################################################################################
# Um Maps zu verwenden, sollen alle Begriffe vereinheitlicht werden.
def keytify(text):
    text = text.strip().lower()
    text = " ".join(text.split())
    return text
################################################################################################





################################################################################################
#semantic information about entities like entity_type and PMDco 3.0.0 subclasses:
patterns = []
entityInformationMap = {}
for _, row in df.iterrows():
    raw = row["entity"]  # Standard
    entity = to_camel_case(row["entity"])
    entitytype =  to_camel_case(row["entity_type"])
    thing = to_camel_case(row["Thing"]) if row["Thing"] else None
    subclass1 = to_camel_case(row["SubClass1"]) if row["SubClass1"] else None
    subclass2 = to_camel_case(row["SubClass2"]) if row["SubClass2"] else None
    subclass3 = to_camel_case(row["SubClass3"]) if row["SubClass3"] else None
    subclass4 = to_camel_case(row["SubClass4"]) if row["SubClass4"] else None
    subclass5 = to_camel_case(row["SubClass5"]) if row["SubClass5"] else None
    subclass6 = to_camel_case(row["SubClass6"]) if row["SubClass6"] else None
    subclass7 = to_camel_case(row["SubClass7"]) if row["SubClass7"] else None
    subclass8 = to_camel_case(row["SubClass8"]) if row["SubClass8"] else None
    subclass9 = to_camel_case(row["SubClass9"]) if row["SubClass9"] else None
    #if "SubClass9" in df.columns: 
    #  subclass9 = to_camel_case(row["SubClass9"])
    #else:
    #  subclass9 = None
    entityInformationMap[keytify(raw)] = [entitytype,entity,subclass1,subclass2,subclass3,subclass4,subclass5,subclass6,subclass7,subclass8,subclass9,thing, None]
    # Mehrere Rechtschreibungen ermöglichen
    nlp.tokenizer.add_special_case(raw, [{ORTH: raw}])
    nlp.tokenizer.add_special_case(raw.lower(), [{ORTH: raw.lower()}])
    nlp.tokenizer.add_special_case(raw.title(), [{ORTH: raw.title()}])
    patterns.append({"label": entitytype, "pattern": [{"LOWER": raw.lower()}]})
################################################################################################





################################################################################################
# Definitions and Labels
df1 = read_ods(definitions_input_dir, 1,
               columns=['entities','entity_types','Definition','Thing','SubClass1','SubClass2',
                        'SubClass3','SubClass4','SubClass5,SubClass6','SubClass7','SubClass8','SubClass9'])  # Einlesen aller Informationen der Definitionen
entity_definition_map = {}
for _, row in df1.iterrows():  # Speichern aller Informationen in einer Map
    raw = row["entities"] if row["entities"] else None
    #print(raw)
    definition = row["Definition"] if row["Definition"] else None
    if raw:
      entity_definition_map[keytify(raw)] = [raw, definition]

ruler.add_patterns(patterns)  # Patterns dem Ruler hinzufügen, die die jeweiligen Entitäten definieren
################################################################################################





################################################################################################
#routines for unit recognition
RANGE_SEP = r'(?:–|—|-|to|and)' # Alles, was einen Range angeben kann
NUM = r'(?:\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)' # Zahl kann 20 oder 20,0 oder 20.0 oder 2.0e+0 sein
UNIT = r'(?:[A-Za-z°µ/%]+)' # Alle möglichen Buchtaben, paar Zeichen, vllt erweitern

## https://regexr.com/ Zum Erstellen der regulären Ausdrücke
#def normalize_units(s):
#    # Für Erkennung von Gradeinheiten
#    s = s.replace('\u25E6', '\u00B0')  # Formatierung für °
#    s = re.sub(r'\bdeg(?:ree)?\s*[cC]\b', '°C', s)  # deg C zu °C
#    s = re.sub(r'(?<=\d)\s*[cC]\b', ' °C', s)  # 500 C zu 500 °C
#    # Versichern, dass Kelvin statt KibiByte erkannt wird
#    s = re.sub(r'(?<=\d)\s*[Kk]\b(?![iI]?B)', ' K', s)
#    # Exponenten wiederherstellen
#    s = re.sub(r'\bm3\b', 'm³', s)
#    s = re.sub(r'\bcm3\b', 'cm³', s)
#    s = re.sub(r'\bmm2\b', 'mm²', s)
#    # Mikrosymbol korrekt erkennen
#    s = s.replace('\u03BC', '\u00B5')
#    s = re.sub(r'\bmicron(s)?\b', 'µm', s, flags=re.I)
#    # Temperatur Ranges korrekt erkennen lassen
#    s = re.sub(r'(\d+(?:\.\d+)?)\s*[–—-]\s*(\d+(?:\.\d+)?)\s*°C',r'\1 °C to \2 °C', s)
#    return s
################################################################################################





################################################################################################
#MAIN ROUTINE

#jump to the directory with the pdfs:
os.chdir(directory)
files=glob.glob("*.pdf")


jsonData=[]
doiList=[]
for f in files:
  print("\nActual file = "+f+"\n")
  #
  #extract the DOI:
  checkDOI=pdf2doi.pdf2doi(f)
  if checkDOI["identifier"]:
    doi=checkDOI["identifier"]
    OpenAlexID=pyalex.Works()["https://doi.org/"+doi]["id"]
    print(OpenAlexID)
  else:
    doi='none'
    OpenAlexID='none'
    print("No doi found!")
  #only if pdf was NOT worked through before:
  if doi in doiList:
    print("Paper  "+f+" was worked through before!")
  else:
    #
    doiList.append(doi)
    #
    #Iterate through the pdf:
    Entities=[]
    EntityTypes=[]
    doc = pymupdf.open(f)
    for page in doc:
      text = page.get_text()
      #print(text)
      #text = normalize_units(text)
      #text = normalize_ranges(text)
      spacy_doc = nlp(text)
      #Durch den Text in 'spacy_doc' durchiterieren:
      for line in spacy_doc.sents:
        ents = []  # Entitätenliste
        #print(line)
        if line.ents:  # Falls Entitäten gefunden wurden
          for i, ent1 in enumerate(line.ents):  # Iterieren durch die Entitätentypen
            #get entity:
            e1 = entityInformationMap.get(keytify(ent1.text))[1]  # Korrekte Schreibweise aus Tabelle
            print(e1)
            #
            #get entity type:
            et1 = entityInformationMap.get(keytify(ent1.text))[0]  # Korrekte Schreibweise aus Tabelle
            #Save in list:
            Entities.append(e1)
            EntityTypes.append(et1)
            #input('')
    #
    #remove duplicates:
    #Entities=list(set(Entities))
    #EntityTypes=list(set(EntityTypes))
    Entities_unique=[]
    EntityTypes_unique=[]
    for ii in range(0,len(Entities)):
      if Entities[ii] not in Entities_unique:
        Entities_unique.append(Entities[ii])
        EntityTypes_unique.append(EntityTypes[ii]) 	
    #Assign:
    Entities=Entities_unique
    EntityTypes=EntityTypes_unique
    #
    print("Found entites: ")
    print(Entities,EntityTypes)
    #
    #relate entities to entity_types:
    #
    #create data list:
    dataList=[{"DOI": doi,"OpenAlexID": OpenAlexID,"Entities": Entities,"Entity_types": EntityTypes}]
    #jsonData = json.loads(dataList)
    #json_formatted_str = json.dumps(jsonData, indent=4)
    #dataList = pprint.pformat(dataList, compact=True).replace("'",'"')
    #jsonData.append(json.dumps(dataList, indent=4))
    jsonData.append(dataList)

with open(fout, 'w') as f_j:
  #json.dump(json_formatted,f_j,ensure_ascii=True,indent=4,sort_keys=True)
  #json.dump(jsonData,f_j,ensure_ascii=True,indent=4,sort_keys=True)
  json.dump(jsonData,f_j,ensure_ascii=False,indent=4,sort_keys=False)
  #pretty_json_str = pprint.pformat(jsonData, compact=True).replace("'",'"')
  #pretty_json_str = pprint.pformat(jsonData, compact=False).replace("'",'"')
  #f_j.write(pretty_json_str)
################################################################################################























