<div align="center">
<img src="./images/OCRFlux.png" alt="OCRFlux Logo" width="300"/>
<hr/>
</div>
<p align="center">
  <a href="https://github.com/chatdoc-com/OCRFlux/blob/main/LICENSE">
    <img alt="GitHub License" src="./images/license.svg" height="20">
  </a>
  <a href="https://github.com/chatdoc-com/OCRFlux/releases">
    <img alt="GitHub release" src="./images/release.svg" height="20">
  </a>
  <a href="https://ocrflux.pdfparser.io/">
    <img alt="Demo" src="./images/demo.svg" height="20">
  </a>
  <a href="https://discord.gg/F33mhsAqqg">
    <img alt="Discord" src="./images/discord.svg" height="20">
  </a>
</p>

OCRFlux is a multimodal large language model based toolkit for converting PDFs and images into clean, readable, plain Markdown text. It aims to push the current state-of-the-art to a significantly higher level.

Try the online demo: [OCRFlux Demo](https://ocrflux.pdfparser.io/)

Functions: **Whole file parsing**
- On each page
    - Convert into text with a natural reading order, even in the presence of multi-column layouts, figures, and insets
    - Support for complicated tables and equations
    - Automatically removes headers and footers

- Cross-page table/paragraph merging
    - Cross-page table merging
    - Cross-page paragraph merging


Key features:
- Superior parsing quality on each page

    It respectively achieves 0.095 higher (from 0.872 to 0.967), 0.109 higher (from 0.858 to 0.967) and 0.187 higher (from 0.780 to 0.967) Edit Distance Similarity (EDS) on our released benchmark [OCRFlux-bench-single](https://huggingface.co/datasets/ChatDOC/OCRFlux-bench-single) than the baseline model [olmOCR-7B-0225-preview](https://huggingface.co/allenai/olmOCR-7B-0225-preview), [Nanonets-OCR-s](https://huggingface.co/nanonets/Nanonets-OCR-s) and [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR).

- Native support for cross-page table/paragraph merging  (to our best this is the first to support this feature in all the open sourced project).

- Based on a 3B parameter VLM, so it can run even on GTX 3090 GPU.

Release:
- [OCRFlux-3B](https://huggingface.co/ChatDOC/OCRFlux-3B) - 3B parameter VLM
- Benchmark for evaluation
    - [OCRFlux-bench-single](https://huggingface.co/datasets/ChatDOC/OCRFlux-bench-single)
    - [OCRFlux-pubtabnet-single](https://huggingface.co/datasets/ChatDOC/OCRFlux-pubtabnet-single)
    - [OCRFlux-bench-cross](https://huggingface.co/datasets/ChatDOC/OCRFlux-bench-cross)
    - [OCRFlux-pubtabnet-cross](https://huggingface.co/datasets/ChatDOC/OCRFlux-pubtabnet-cross)


### News
 - Jun 17, 2025 - v0.1.0 -  Initial public launch and demo.

### Benchmark for single-page parsing

We ship two comprehensive benchmarks to help measure the performance of our OCR system in single-page parsing:

  - [OCRFlux-bench-single](https://huggingface.co/datasets/ChatDOC/OCRFlux-bench-single): Containing 2000 pdf pages (1000 English pages and 1000 Chinese pages) and their ground-truth Markdowns (manually labeled with multi-round check).

  - [OCRFlux-pubtabnet-single](https://huggingface.co/datasets/ChatDOC/OCRFlux-pubtabnet-single): Derived from the public [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) benchmark with some format transformation. It contains 9064 HTML table samples, which are split into simple tables and complex tables according to whether they have rowspan and colspan cells.

We emphasize that the released benchmarks are NOT included in our training and evaluation data. The following is the main result:


1. In [OCRFlux-bench-single](https://huggingface.co/datasets/ChatDOC/OCRFlux-bench-single), we calculated the Edit Distance Similarity (EDS) between the generated Markdowns and the ground-truth Markdowns as the metric.

    <table>
      <thead>
        <tr>
          <th>Language</th>
          <th>Model</th>
          <th>Avg EDS ↑</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td rowspan="4">English</td>
          <td>olmOCR-7B-0225-preview</td>
          <td>0.885</td>
        </tr>
        <tr>
          <td>Nanonets-OCR-s</td>
          <td>0.870</td>
        </tr>
        <tr>
          <td>MonkeyOCR</td>
          <td>0.828</td>
        </tr>
        <tr>
          <td><strong><a href="https://huggingface.co/ChatDOC/OCRFlux-3B">OCRFlux-3B</a></strong></td>
          <td>0.971</td>
        </tr>
        <tr>
          <td rowspan="4">Chinese</td>
          <td>olmOCR-7B-0225-preview</td>
          <td>0.859</td>
        </tr>
        <tr>
          <td>Nanonets-OCR-s</td>
          <td>0.846</td>
        </tr>
        <tr>
          <td>MonkeyOCR</td>
          <td>0.731</td>
        </tr>
        <tr>
          <td><strong><a href="https://huggingface.co/ChatDOC/OCRFlux-3B">OCRFlux-3B</a></strong></td>
          <td>0.962</td>
        </tr>
        <tr>
          <td rowspan="4">Total</td>
          <td>olmOCR-7B-0225-preview</td>
          <td>0.872</td>
        </tr>
        <tr>
          <td>Nanonets-OCR-s</td>
          <td>0.858</td>
        </tr>
        <tr>
          <td>MonkeyOCR</td>
          <td>0.780</td>
        </tr>
        <tr>
          <td><strong><a href="https://huggingface.co/ChatDOC/OCRFlux-3B">OCRFlux-3B</a></strong></td>
          <td>0.967</td>
        </tr>
      </tbody>
    </table>

2. In [OCRFlux-pubtabnet-single](https://huggingface.co/datasets/ChatDOC/OCRFlux-pubtabnet-single), we calculated the Tree Edit Distance-based Similarity (TEDS) between the generated HTML tables and the ground-truth HTML tables as the metric.
    <table>
      <thead>
        <tr>
          <th>Type</th>
          <th>Model</th>
          <th>Avg TEDS ↑</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td rowspan="4">Simple</td>
          <td>olmOCR-7B-0225-preview</td>
          <td>0.810</td>
        </tr>
        <tr>
          <td>Nanonets-OCR-s</td>
          <td>0.882</td>
        </tr>
        <tr>
          <td>MonkeyOCR</td>
          <td>0.880</td>
        </tr>
        <tr>
          <td><strong><a href="https://huggingface.co/ChatDOC/OCRFlux-3B">OCRFlux-3B</a></strong></td>
          <td>0.912</td>
        </tr>
        <tr>
          <td rowspan="4">Complex</td>
          <td>olmOCR-7B-0225-preview</td>
          <td>0.676</td>
        </tr>
        <tr>
          <td>Nanonets-OCR-s</td>
          <td>0.772</td>
        </tr>
        <tr>
          <td><strong>MonkeyOCR<strong></td>
          <td>0.826</td>
        </tr>
        <tr>
          <td><a href="https://huggingface.co/ChatDOC/OCRFlux-3B">OCRFlux-3B</a></td>
          <td>0.807</td>
        </tr>
        <tr>
          <td rowspan="4">Total</td>
          <td>olmOCR-7B-0225-preview</td>
          <td>0.744</td>
        </tr>
        <tr>
          <td>Nanonets-OCR-s</td>
          <td>0.828</td>
        </tr>
        <tr>
          <td>MonkeyOCR</td>
          <td>0.853</td>
        </tr>
        <tr>
          <td><strong><a href="https://huggingface.co/ChatDOC/OCRFlux-3B">OCRFlux-3B</a></strong></td>
          <td>0.861</td>
        </tr>
      </tbody>
    </table>

We also conduct some case studies to show the superiority of our model in the [blog](https://ocrflux.pdfparser.io/#/blog) article.

### Benchmark for cross-page table/paragraph merging

PDF documents are typically paginated, which often results in tables or paragraphs being split across consecutive pages. Accurately detecting and merging such cross-page structures is crucial to avoid generating incomplete or fragmented content. 

The detection task can be formulated as follows: given the Markdowns of two consecutive pages—each structured as a list of Markdown elements (e.g., paragraphs and tables)—the goal is to identify the indexes of elements that should be merged across the pages.

Then for the merging task, if the elements to be merged are paragraphs, we can just concate them. However, for two table fragments, their merging is much more challenging. For example, the table spanning multiple pages will repeat the header of the first page on the second page. Another difficult scenario is that the table cell contains long content that spans multiple lines within the cell, with the first few lines appearing on the previous page and the remaining lines continuing on the next page. We also observe some cases where tables with a large number of columns are split vertically and placed on two consecutive pages. More examples of cross-page tables can be found in our [blog](https://ocrflux.pdfparser.io/#/blog) article. To address these issues, we develop the LLM model for cross-page table merging. Specifically, this model takes two split table fragments as input and generates a complete, well-structured table as output.

We ship two comprehensive benchmarks to help measure the performance of our OCR system in cross-page table/paragraph detection and merging tasks respectively:

  - [OCRFlux-bench-cross](https://huggingface.co/datasets/ChatDOC/OCRFlux-bench-cross): Containing 1000 samples (500 English samples and 500 Chinese samples), each sample contains the Markdown element lists of two consecutive pages, along with the indexes of elements that need to be merged (manually labeled through multiple rounds of review). If no tables or paragraphs require merging, the indexes in the annotation data are left empty.

  - [OCRFlux-pubtabnet-cross](https://huggingface.co/datasets/ChatDOC/OCRFlux-pubtabnet-cross): Containing 9064 pairs of split table fragments, along with their corresponding ground-truth merged versions.

The released benchmarks are NOT included in our training and evaluation data neither. The following is the main result:

1. In [OCRFlux-bench-cross](https://huggingface.co/datasets/ChatDOC/OCRFlux-bench-cross), we caculated the Accuracy, Precision, Recall and F1 score as the metric. Notice that the detection results are right only when it accurately judges whether there are elements that need to be merged across the two pages and output the right indexes of them.

    | Language | Precision ↑ | Recall ↑ | F1 ↑  | Accuracy ↑ |
    |----------|-------------|----------|-------|------------|
    | English  | 0.992       | 0.964    | 0.978 | 0.978      |
    | Chinese  | 1.000       | 0.988    | 0.994 | 0.994      |
    | Total    | 0.996       | 0.976    | 0.986 | 0.986      |

2. In [OCRFlux-pubtabnet-cross](https://huggingface.co/datasets/ChatDOC/OCRFlux-pubtabnet-cross), we calculate the Tree Edit Distance-based Similarity (TEDS) between the generated merged table and the ground-truth merged table as the metric.

    | Table type | Avg TEDS ↑   |
    |------------|--------------|
    | Simple     | 0.965        |
    | Complex    | 0.935        |
    | Total      | 0.950        |

### Installation

Requirements:
 - Recent NVIDIA GPU (tested on RTX 3090, 4090, L40S, A100, H100) with at least 12 GB of GPU RAM
 - 20GB of free disk space

You will need to install poppler-utils and additional fonts for rendering PDF images.

Install dependencies (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install poppler-utils poppler-data ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools
```

Set up a conda environment and install OCRFlux. The requirements for running OCRFlux
are difficult to install in an existing python environment, so please do make a clean python environment to install into.
```bash
conda create -n ocrflux python=3.11
conda activate ocrflux

git clone https://github.com/chatdoc-com/OCRFlux.git
cd OCRFlux

pip install -e . --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/
```

### Local Usage Example

For quick testing, try the [web demo](https://5f65ccdc2d4fd2f364.gradio.live). To run locally, a GPU is required, as inference is powered by [vllm](hhttps://github.com/vllm-project/vllm) under the hood.

- For a pdf document:
    ```bash
    python -m ocrflux.pipeline ./localworkspace --data test.pdf --model /model_dir/OCRFlux-3B
    ```

- For an image:
    ```bash
    python -m ocrflux.pipeline ./localworkspace --data test_page.png --model /model_dir/OCRFlux-3B
    ```

- For a directory of pdf or images:
    ```bash
    python -m ocrflux.pipeline ./localworkspace --data test_pdf_dir/* --model /model_dir/OCRFlux-3B
    ```
You can set `--skip_cross_page_merge` to skip the cross-page merging in the parsing process to accelerate, it would simply concatenate the parsing results of each page to generate final Markdown of the document.

Results will be stored as JSONL files in the `./localworkspace/results` directory. 

Each line in JSONL files is a json object with the following fields:

```
{
    "orig_path": str,  # the path to the raw pdf or image file
    "num_pages": int,  # the number of pages in the pdf file
    "document_text": str, # the Markdown text of the converted pdf or image file
    "page_texts": dict, # the Markdown texts of each page in the pdf file, the key is the page index and the value is the Markdown text of the page
    "fallback_pages": [int], # the page indexes that are not converted successfully
}
```

### API for directly calling OCRFlux (New)
You can use the inference API to directly call OCRFlux in your codes without using an online vllm server like following:

```
from vllm import LLM
from ocrflux.inference import parse

file_path = 'test.pdf'
# file_path = 'test.png'
llm = LLM(model="model_dir/OCRFlux-3B",gpu_memory_utilization=0.8,max_model_len=8192)
result = parse(llm,file_path)
if result != None:
    document_markdown = result['document_text']
    print(document_markdown)
    with open('test.md','w') as f:
        f.write(document_markdown)
else:
    print("Parse failed.")
```
If parsing is failed or there are fallback pages in the result, you can try to set the argument `max_page_retries` for the `parse` function with a positive integer to get a better result. But it may cause longer inference time.

### Local Deployment (New)

Run the following command to start the server:

```bash
bash ocrflux/server.sh /path/to/model port
```

For example, the following command:

```bash
bash ocrflux/server.sh ChatDOC/OCRFlux-3B 30024
```

It will start a vllm server on port 30024. You can also start server by yourself using other methods like `sglang`.

After the server is started, you can use the `request` api to request it to parse a pdf file or an image file like following:

```
import asyncio
from argparse import Namespace
from ocrflux.client import request
args = Namespace(
    model="/path/to/OCRFlux-3B",
    skip_cross_page_merge=False,
    max_page_retries=1,
    url="http://localhost",
    port=30024,
)
file_path = 'test.pdf'
# file_path = 'test.png'
result = asyncio.run(request(args,file_path))
if result != None:
    document_markdown = result['document_text']
    print(document_markdown)
    with open('test.md','w') as f:
        f.write(document_markdown)
else:
    print("Parse failed.")
```


### Docker Usage

Requirements:

- Docker with GPU support [(NVIDIA Toolkit)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Pre-downloaded model: [OCRFlux-3B](https://huggingface.co/ChatDOC/OCRFlux-3B)

To use OCRFlux in a docker container, you can use the following example command to start the docker container firstly:

```bash
docker run -it --gpus all \
  -v /path/to/localworkspace:/localworkspace \
  -v /path/to/test_pdf_dir:/test_pdf_dir \
  -v /path/to/OCRFlux-3B:/OCRFlux-3B \
  --entrypoint bash \
  chatdoc/ocrflux:latest
```

and then run the following command on the docker container to parse document files:

```bash
python3.12 -m ocrflux.pipeline /localworkspace/ocrflux_results --data /test_pdf_dir/*.pdf --model /OCRFlux-3B/
```

The parsing results will be stored in `/localworkspace/ocrflux_results` directory. Run the following command to generate the final Markdown files:
```bash
python -m ocrflux.jsonl_to_markdown ./localworkspace/ocrflux_results
```

#### Viewing Results
Generate the final Markdown files by running the following command. Generated Markdown files will be in `./localworkspace/markdowns/DOCUMENT_NAME` directory.

```bash
python -m ocrflux.jsonl_to_markdown ./localworkspace
```

### Full documentation for the pipeline

```bash
python -m ocrflux.pipeline --help
usage: pipeline.py [-h] [--task {pdf2markdown,merge_pages,merge_tables}] [--data [DATA ...]] [--pages_per_group PAGES_PER_GROUP] [--max_page_retries MAX_PAGE_RETRIES]
                   [--max_page_error_rate MAX_PAGE_ERROR_RATE] [--workers WORKERS] [--model MODEL] [--model_max_context MODEL_MAX_CONTEXT] [--model_chat_template MODEL_CHAT_TEMPLATE]
                   [--target_longest_image_dim TARGET_LONGEST_IMAGE_DIM] [--skip_cross_page_merge] [--port PORT]
                   workspace

Manager for running millions of PDFs through a batch inference pipeline

positional arguments:
  workspace             The filesystem path where work will be stored, can be a local folder

options:
  -h, --help            show this help message and exit
  --data [DATA ...]     List of paths to files to process
  --pages_per_group PAGES_PER_GROUP
                        Aiming for this many pdf pages per work item group
  --max_page_retries MAX_PAGE_RETRIES
                        Max number of times we will retry rendering a page
  --max_page_error_rate MAX_PAGE_ERROR_RATE
                        Rate of allowable failed pages in a document, 1/250 by default
  --workers WORKERS     Number of workers to run at a time
  --model MODEL         The path to the model
  --model_max_context MODEL_MAX_CONTEXT
                        Maximum context length that the model was fine tuned under
  --model_chat_template MODEL_CHAT_TEMPLATE
                        Chat template to pass to vllm server
  --target_longest_image_dim TARGET_LONGEST_IMAGE_DIM
                        Dimension on longest side to use for rendering the pdf pages
  --skip_cross_page_merge
                        Whether to skip cross-page merging
  --port PORT           Port to use for the VLLM server
```

## Code overview

There are some nice reusable pieces of the code that may be useful for your own projects:
 - Processing millions of PDFs through our released model using VLLM - [pipeline.py](https://github.com/chatdoc-com/OCRFlux/blob/main/ocrflux/pipeline.py)
 - Generating final Markdowns from jsonl files - [jsonl_to_markdown.py](https://github.com/chatdoc-com/OCRFlux/blob/main/ocrflux/jsonl_to_markdown.py)
 - Evaluating the model on the single-page parsing task - [eval_page_to_markdown.py](https://github.com/chatdoc-com/OCRFlux/blob/main/eval/eval_page_to_markdown.py)
 - Evaluating the model on the table parising task - [eval_table_to_html.py](https://github.com/chatdoc-com/OCRFlux/blob/main/eval/eval_table_to_html.py)
 - Evaluating the model on the paragraphs/tables merging detection task - [eval_element_merge_detect.py](https://github.com/chatdoc-com/OCRFlux/blob/main/eval/eval_element_merge_detect.py)
 - Evaluating the model on the table merging task - [eval_html_table_merge.py](https://github.com/chatdoc-com/OCRFlux/blob/main/eval/eval_html_table_merge.py)


## Team

<!-- start team -->

**OCRFlux** is developed and maintained by the ChatDOC team, backed by [ChatDOC](https://chatdoc.com/).

<!-- end team -->

## License

<!-- start license -->

**OCRFlux** is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
A full copy of the license can be found [on GitHub](https://github.com/allenai/OCRFlux/blob/main/LICENSE).

<!-- end license -->
