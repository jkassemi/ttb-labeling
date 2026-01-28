# COLA Label Verification

## Architecture

A Qwen/Qwen2.5-VL-3B-Instruct vision language model is used to extract label details and validated against extracted data from Paddle OCR. Labels are queued for processing and presented to the verifier via a prototype Gradio interface (SSE) without reload. A simple rules engine is used to check the label against business requirements and compare against application data if provided.

The project is built with Python, with requirements managed by uv. It manages a simple in-process queue. Local models to avoid external services.

## Usage

Application deployed (temporarily) at https://ttb.kassemi.org/.

### Application Submissions

1. Upload label images for an application to the "Label images" file selector.
2. If matching is desired, fill the optional application fields. 
3. Select Queue verification. The application will appear in the "Processing queue" section.

### Application Verification

1. Click an item in the "Review queue" after it's completed processing.
2. Review the Checklist findings for findings that need review and review the message.
3. Choose to Accept or Deny.

**Note: Accepting or Denying will just remove the item from the queue. No further processing/persistence is in place**

## Limitations

* Docker integration pending
* In-process queue, not cross-restart persist or horizontally scalable
* Gradio for rapid UX prototyping limits interface
* Bold detection capabilities in government warning limited
* No application/approver RBAC or separate interfaces
* No integration with COLA systems
* Human in the loop - Jenny presents adherence to the requirements as rigid and strict, but given Dave's experience I'm assuming a human in the loop is essential to ensure an appropriate level of nuance is applied.

## Assumptions

* GPU-focused development is acceptable - GPU via Azure container apps is a potential deployment target.
* Gradio is sufficient for the demonstration, but a simpler interface could be developed based on user research.
* Authentication and role-based access are not within scope. 
* Sarah's <5s performance requirement is likely based more on the UX than the actual processing speed. The primary means of addressing it within this project is through background processing - presenting applications to the reviewer *after* they've been processed, not requiring the reviewer to submit and wait for each one.
* Integration directly with COLA is not in scope (and would lead to potentially significant input changes).
* Government warning exactness: if the full statement matches the canonical warning text but is rendered in all caps, we treat it as compliant (OCR often returns all-caps text).

## Install

Requires Python 3.12+ and a moderately sized GPU. This was tested on a system with a NVIDIA RTX 4000 (~20GiB) at around 50% capacity. Install the dependencies with:

```bash
uv sync
```

## Run (Gradio Demo)

Launch the Gradio UI:

```bash
uv run python -m cola_label_verification.gradio_app
```

Visit the local address printed on the screen and follow usage instructions.

## Potential Questions

1. What benefit do we get from up-front government verification of labels? Is there an upstream (legislative) solution to this problem? Can we make it reactive (courts handle issues after the fact)? Private labeling solutions?
2. Sarah Chen mentioned a vendor had a solution that worked, but took too long to process individual labels. Depending on the maturity, complexity, and cost of that solution, it might be worth reconsidering along with a few minor UX changes to enable background processing of applications.
