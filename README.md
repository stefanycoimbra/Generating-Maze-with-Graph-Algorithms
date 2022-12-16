# Generating-Maze-with-Graph-Algorithms
Gerador de labirintos utilizando os algoritmos de grafos:
- Binary Tree
- Kruskal

Depois de gerado o labirinto, sua solução é dada pelo algoritmo:
- Dijkstra

## Caso utilizado Windows, para compilar o arquivo em Python:
- Baixar arquivo 64-bit do PyOpenGL e PyOpenGL accelerate do site: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl
- Para escolher a versão correta, rodar no prompt de comando: "python --version"

Por exemplo, se a versão do Python for a 3.10, então o download deve ser esses dois arquivos:
```bash
PyOpenGL-3.1.6-cp310-cp310-win_amd64.whl
PyOpenGL_accelerate-3.1.6-cp310-cp310-win_amd64.whl
```
Na pasta onde foram feitos os downloads, no prompt de comando, rodar:
```bash
pip install PyOpenGL-3.1.6-cp310-cp310-win_amd64.whl --force-reinstall
pip install PyOpenGL_accelerate-3.1.6-cp310-cp310-win_amd64.whl --force-reinstall
```

## Execução
Na pasta com o arquivo `maze.py`, rodar: `.python maze.py`

## Controles
- `w`ou `W`: afasta a câmera de observação do labirinto.
- `s`ou `S`: aproxima a câmera de observação do labirinto.
- `a` ou `A`: rotaciona labirinto para a esquerda em torno do eixo y.
- `d` ou `D`: rotaciona labirinto para a direita em torno do eixo y.
- `t` ou `T`: geração do labirinto pelo algoritmo de árvore binária.
- `k` ou `K`: geração do labirinto pelo algoritmo de Kruskal.
- `q`: sair do programa.

## Time
* Stéfany Coura Coimbra (<img src="https://img.icons8.com/ios-glyphs/30/000000/github.png"/> https://github.com/stefanycoimbra)
* Viviane Cardosina Cordeiro (<img src="https://img.icons8.com/ios-glyphs/30/000000/github.png"/> https://github.com/VivianeCordeiro)
* Ytalo Ysmaicon Gomes (<img src="https://img.icons8.com/ios-glyphs/30/000000/github.png"/> https://github.com/ysmaicon)
