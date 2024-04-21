
# PSA
psa Score prediction by fine-tuned ERes2net model

 
## Getting started
The package installation was tested with python3.9

```bash
pip install git+https://github.com/zdpBuilder/psa-main
```
## Inference

```python
from PSA import get_PSA
model =  get_PSA()

PSA_score = model.get_similarity_result_by_MI("path/to/dir/with/wav/files","path/to/dir/with/wav/files")

```

## Citation and Acknowledgment
