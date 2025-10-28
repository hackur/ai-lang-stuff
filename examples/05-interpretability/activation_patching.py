"""
Activation Patching with TransformerLens

This example demonstrates activation patching (also called causal interventions)
to understand which model components are causally responsible for specific behaviors.

Activation patching involves:
1. Running model on a "clean" prompt
2. Running model on a "corrupted" prompt
3. Patching activations from clean run into corrupted run
4. Measuring how much behavior is restored

This helps identify which components (layers, heads, neurons) are critical
for specific model behaviors.

Prerequisites:
    uv add transformer-lens torch numpy

Usage:
    python activation_patching.py

References:
    - https://transformerlensorg.github.io/TransformerLens/
    - Anthropic's "Interpretability in the Wild" paper
"""

import torch
import numpy as np
from typing import Dict
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import matplotlib.pyplot as plt


def setup_model(model_name: str = "gpt2-small") -> HookedTransformer:
    """
    Load a model with TransformerLens hooks.

    Args:
        model_name: Name of the model to load

    Returns:
        HookedTransformer model
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Loading {model_name} on {device}...")

    model = HookedTransformer.from_pretrained(
        model_name, center_unembed=True, center_writing_weights=True, fold_ln=True, device=device
    )

    print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads per layer")
    return model


def get_logit_diff(logits: torch.Tensor, correct_token_id: int, incorrect_token_id: int) -> float:
    """
    Compute difference in logits between correct and incorrect answer.

    This is our metric for how well the model predicts the correct answer.
    Higher = model prefers correct answer more.

    Args:
        logits: Model output logits [batch, seq_len, vocab]
        correct_token_id: ID of the correct next token
        incorrect_token_id: ID of an incorrect alternative

    Returns:
        Difference in logits (correct - incorrect)
    """
    # Get final position logits
    final_logits = logits[0, -1, :]

    correct_logit = final_logits[correct_token_id].item()
    incorrect_logit = final_logits[incorrect_token_id].item()

    return correct_logit - incorrect_logit


def patch_activation(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    clean_cache: dict,
    patch_layer: int,
    patch_head: int = None,
    component: str = "attn_out",
) -> float:
    """
    Patch a specific component from clean run into corrupted run.

    This tests: "If we fix this component to its clean value,
    how much does the model's behavior improve?"

    Args:
        model: HookedTransformer model
        clean_tokens: Tokens for clean (correct) prompt
        corrupted_tokens: Tokens for corrupted (incorrect) prompt
        clean_cache: Cached activations from clean run
        patch_layer: Which layer to patch
        patch_head: Which head to patch (None = patch entire layer)
        component: Which component to patch ('attn_out', 'mlp_out', 'resid_post')

    Returns:
        Logit difference after patching
    """

    # Define the patch hook
    def patch_hook(activation, hook):
        """Replace activation with clean version."""
        if patch_head is None:
            # Patch entire layer
            activation[:] = clean_cache[hook.name]
        else:
            # Patch specific head
            # Attention output shape: [batch, seq, n_heads, d_head]
            activation[:, :, patch_head, :] = clean_cache[hook.name][:, :, patch_head, :]
        return activation

    # Get activation name
    if component == "attn_out":
        act_name = get_act_name("z", patch_layer)  # Attention output
    elif component == "mlp_out":
        act_name = get_act_name("mlp_out", patch_layer)
    elif component == "resid_post":
        act_name = get_act_name("resid_post", patch_layer)
    else:
        raise ValueError(f"Unknown component: {component}")

    # Run corrupted prompt with patching
    logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(act_name, patch_hook)])

    return logits


def run_patching_experiment(
    model: HookedTransformer,
    clean_prompt: str,
    corrupted_prompt: str,
    correct_answer: str,
    incorrect_answer: str,
) -> Dict[str, np.ndarray]:
    """
    Run full activation patching experiment across all layers and heads.

    Args:
        model: HookedTransformer model
        clean_prompt: Prompt that leads to correct answer
        corrupted_prompt: Prompt that leads to incorrect answer
        correct_answer: Expected correct completion
        incorrect_answer: Alternative incorrect completion

    Returns:
        Dictionary mapping component types to patching results
    """
    print("\n" + "=" * 70)
    print("ACTIVATION PATCHING EXPERIMENT")
    print("=" * 70)

    # Tokenize
    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)
    correct_token_id = model.to_single_token(correct_answer)
    incorrect_token_id = model.to_single_token(incorrect_answer)

    print(f"\nClean prompt: {clean_prompt}")
    print(f"Corrupted prompt: {corrupted_prompt}")
    print(f"Correct answer: '{correct_answer}' (token {correct_token_id})")
    print(f"Incorrect answer: '{incorrect_answer}' (token {incorrect_token_id})")

    # Run clean and corrupted baselines
    print("\n1. Running baseline computations...")
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

    clean_diff = get_logit_diff(clean_logits, correct_token_id, incorrect_token_id)
    corrupted_diff = get_logit_diff(corrupted_logits, correct_token_id, incorrect_token_id)

    print(f"   Clean logit diff: {clean_diff:.3f} (prefers correct answer)")
    print(f"   Corrupted logit diff: {corrupted_diff:.3f}")

    # Patch each layer and head
    print("\n2. Patching attention heads...")
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Results storage: [layer, head]
    attn_patching_results = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        print(f"   Layer {layer}/{n_layers-1}", end="\r")
        for head in range(n_heads):
            # Patch this head
            patched_logits = patch_activation(
                model,
                clean_tokens,
                corrupted_tokens,
                clean_cache,
                patch_layer=layer,
                patch_head=head,
                component="attn_out",
            )

            patched_diff = get_logit_diff(patched_logits, correct_token_id, incorrect_token_id)

            # Store normalized recovery: what fraction of the gap did we recover?
            # 0 = no effect, 1 = full recovery to clean performance
            if abs(clean_diff - corrupted_diff) > 1e-6:
                recovery = (patched_diff - corrupted_diff) / (clean_diff - corrupted_diff)
            else:
                recovery = 0.0

            attn_patching_results[layer, head] = recovery

    print(f"   Layer {n_layers-1}/{n_layers-1} ✓")

    # Patch MLP layers
    print("\n3. Patching MLP layers...")
    mlp_patching_results = np.zeros(n_layers)

    for layer in range(n_layers):
        print(f"   Layer {layer}/{n_layers-1}", end="\r")

        patched_logits = patch_activation(
            model,
            clean_tokens,
            corrupted_tokens,
            clean_cache,
            patch_layer=layer,
            component="mlp_out",
        )

        patched_diff = get_logit_diff(patched_logits, correct_token_id, incorrect_token_id)

        if abs(clean_diff - corrupted_diff) > 1e-6:
            recovery = (patched_diff - corrupted_diff) / (clean_diff - corrupted_diff)
        else:
            recovery = 0.0

        mlp_patching_results[layer] = recovery

    print(f"   Layer {n_layers-1}/{n_layers-1} ✓")

    print("\n" + "=" * 70)

    return {
        "attention": attn_patching_results,
        "mlp": mlp_patching_results,
        "clean_diff": clean_diff,
        "corrupted_diff": corrupted_diff,
    }


def visualize_results(results: Dict[str, np.ndarray], save_path: str = None):
    """
    Visualize activation patching results.

    Args:
        results: Dictionary from run_patching_experiment
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot attention head patching
    im1 = axes[0].imshow(results["attention"], cmap="RdYlGn", vmin=-0.5, vmax=1.0, aspect="auto")
    axes[0].set_title("Attention Head Patching Results", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Head", fontsize=12)
    axes[0].set_ylabel("Layer", fontsize=12)
    axes[0].set_xticks(range(results["attention"].shape[1]))
    axes[0].set_yticks(range(results["attention"].shape[0]))
    plt.colorbar(im1, ax=axes[0], label="Behavior Recovery (0=none, 1=full)")

    # Annotate highest values
    for layer in range(results["attention"].shape[0]):
        for head in range(results["attention"].shape[1]):
            value = results["attention"][layer, head]
            if value > 0.5:  # Only annotate high values
                axes[0].text(
                    head,
                    layer,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black" if value < 0.7 else "white",
                    fontsize=8,
                )

    # Plot MLP patching
    axes[1].barh(range(len(results["mlp"])), results["mlp"], color="steelblue")
    axes[1].set_title("MLP Layer Patching Results", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Behavior Recovery", fontsize=12)
    axes[1].set_ylabel("Layer", fontsize=12)
    axes[1].set_yticks(range(len(results["mlp"])))
    axes[1].axvline(x=0, color="black", linestyle="--", linewidth=0.5)
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")

    plt.show()


def print_top_components(results: Dict[str, np.ndarray], top_k: int = 5):
    """
    Print the most important components identified by patching.

    Args:
        results: Dictionary from run_patching_experiment
        top_k: Number of top components to show
    """
    print("\n" + "=" * 70)
    print("TOP CAUSAL COMPONENTS")
    print("=" * 70)

    # Top attention heads
    print(f"\nTop {top_k} Attention Heads:")
    print("-" * 40)
    attn_flat = results["attention"].flatten()
    attn_indices = np.argsort(attn_flat)[::-1][:top_k]

    for i, idx in enumerate(attn_indices, 1):
        layer = idx // results["attention"].shape[1]
        head = idx % results["attention"].shape[1]
        score = results["attention"][layer, head]
        print(f"{i}. Layer {layer:2d}, Head {head:2d}: {score:6.3f} recovery")

    # Top MLP layers
    print(f"\nTop {top_k} MLP Layers:")
    print("-" * 40)
    mlp_indices = np.argsort(results["mlp"])[::-1][:top_k]

    for i, layer in enumerate(mlp_indices, 1):
        score = results["mlp"][layer]
        print(f"{i}. Layer {layer:2d}: {score:6.3f} recovery")

    print("\n" + "=" * 70)


def main():
    """Run activation patching demonstrations."""

    # Load model
    model = setup_model("gpt2-small")

    # Example 1: Indirect Object Identification (IOI)
    # Task: "When Mary and John went to the store, Mary gave a drink to" → " John"
    # Test if model can identify the indirect object (second name)

    print("\n\n" + "=" * 70)
    print("EXAMPLE 1: INDIRECT OBJECT IDENTIFICATION")
    print("=" * 70)
    print("\nTask: Predict the second name in a sentence.")
    print("Clean: 'Mary and John... Mary gave to' → should predict 'John'")
    print("Corrupted: Names swapped or randomized")

    clean_prompt = "When Mary and John went to the store, Mary gave a drink to"
    corrupted_prompt = "When John and Mary went to the store, John gave a drink to"

    results_ioi = run_patching_experiment(
        model,
        clean_prompt=clean_prompt,
        corrupted_prompt=corrupted_prompt,
        correct_answer=" John",
        incorrect_answer=" Mary",
    )

    print_top_components(results_ioi, top_k=5)
    visualize_results(results_ioi, save_path="ioi_patching_results.png")

    # Example 2: Factual Recall
    # Task: "The Eiffel Tower is located in" → " Paris"

    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: FACTUAL RECALL")
    print("=" * 70)
    print("\nTask: Recall factual knowledge")
    print("Clean: 'The Eiffel Tower is in' → should predict 'Paris'")
    print("Corrupted: Different landmark")

    clean_prompt = "The Eiffel Tower is located in"
    corrupted_prompt = "The Statue of Liberty is located in"

    results_fact = run_patching_experiment(
        model,
        clean_prompt=clean_prompt,
        corrupted_prompt=corrupted_prompt,
        correct_answer=" Paris",
        incorrect_answer=" New",
    )

    print_top_components(results_fact, top_k=5)
    visualize_results(results_fact, save_path="factual_patching_results.png")

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY & INTERPRETATION")
    print("=" * 70)
    print(
        """
Key Insights from Activation Patching:

1. CAUSAL vs CORRELATIONAL:
   - High patching score = component is CAUSALLY important
   - Just looking at activations shows correlation, not causation
   - Patching reveals what's necessary for the behavior

2. ATTENTION HEADS:
   - Different heads specialize in different tasks
   - IOI often involves heads in layers 7-9 (in GPT-2)
   - Name mover heads copy the correct name to output
   - Duplicate token heads identify repeated entities

3. MLP LAYERS:
   - MLPs often store factual knowledge
   - Later layers tend to be more task-specific
   - Early layers handle more general processing

4. NEXT STEPS:
   - Try path patching to find circuits (chains of components)
   - Use attribution patching for more precise localization
   - Investigate what makes these components work (circuit analysis)

5. PRACTICAL APPLICATIONS:
   - Model editing: modify specific behaviors
   - Interpretability: understand model internals
   - Debugging: find where errors originate
   - Safety: identify concerning capabilities
    """
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
