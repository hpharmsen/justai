---
title: In een worktree draait je voorbeeld de code van main
date: 2026-09-26
tags: [worktree, verificatie, sys-path, editable-install, examples]
files:
  - examples/system_one.py
  - venv/lib/python3.13/site-packages/__editable___justai_4_2_2_finder.py
---

# In een worktree draait je voorbeeld de code van main

## Aanleiding

`Model.classify` was af in de worktree van `/hp:build`. Het voorbeeld ernaast viel om met
`AttributeError: 'Model' object has no attribute 'classify'`, terwijl de methode er
gewoon stond en de hele testsuite groen was.

## De kern

**Een script zet zijn eigen map vooraan op `sys.path`, niet je werkmap.** Gemeten op deze
machine:

```
venv/bin/python -c "import sys; print(sys.path[0])"   →  ''                 (cwd)
venv/bin/python pad/naar/script.py                    →  'pad/naar'         (scriptmap)
```

Bij `venv/bin/python examples/system_one.py` is `sys.path[0]` dus `examples/`. Daar staat
geen `justai/`. De import valt door naar de editable install, en die wijst naar
`/Users/hp/proj/justai/justai`. Dat is de hoofdmap, niet de worktree.

Let op wat daar misgaat. Niet de import faalt, de *verkeerde* import slaagt. In dit geval
gaf dat een luide `AttributeError`, want `classify` bestond in main nog niet. Was de
methode er wel geweest en alleen anders, dan had het voorbeeld keurig gedraaid en het
gedrag van main laten zien. Je verifieert dan iets wat je niet gebouwd hebt.

Dat raakt `/hp:build` recht in de verificatiestap. Die vraagt om tests plus een demo, en
juist de demo is het stuk dat stil de oude code pakt. Tests niet, want pytest zet de
rootdir vooraan, dus die kijken wel naar de worktree. Precies die asymmetrie maakt het
verraderlijk: groen op alles, behalve dat de demo een ander pakket draaide.

## Wat je doet

Draai een voorbeeld in een worktree met de worktree op het pad:

```bash
cd .worktrees/<branch>
PYTHONPATH=$PWD venv/bin/python examples/<naam>.py
```

Twijfel je, vraag het pakket waar het vandaan komt. Eén regel, en je weet het:

```python
import justai; print(justai.__file__)
```

Na de merge is `PYTHONPATH` niet meer nodig, want dan is de hoofdmap ook de juiste map.

## Toepasbaarheid

Geldt voor elk script in `examples/` en voor elk los debugscript dat je in een worktree
neerzet. Niet voor `pytest` en niet voor `python -c`, die hebben de werkmap al vooraan.
