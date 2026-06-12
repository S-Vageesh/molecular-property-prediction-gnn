"""Generate and save 2D molecule images from SMILES strings.

This module is a standalone utility that sits alongside the GCN prediction
pipeline. It does not touch model weights, training logic, or graph
featurization — its only job is to turn a SMILES string into a PNG file.

Typical usage
-------------
    from molecule_visualizer import visualize_molecule

    image_path = visualize_molecule("CCO")
    print(image_path)  # generated_molecules/CCO.png
"""

from __future__ import annotations

import re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw


# Output directory, relative to the project root (one level above src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "generated_molecules"

# Rendered image dimensions in pixels (width × height).
IMAGE_SIZE = (400, 400)


def _smiles_to_filename(smiles: str) -> str:
    """Derive a safe filesystem filename from a canonical SMILES string.

    SMILES strings can contain characters that are illegal or ambiguous in
    file paths on common operating systems (e.g. '/', '\\', ':', '*', '?').
    This function replaces every non-alphanumeric character (except hyphens
    and underscores) with an underscore so the result is always a valid,
    human-readable filename without path separators or shell-special chars.

    Parameters
    ----------
    smiles:
        A canonical SMILES string produced by RDKit (already sanitized).

    Returns
    -------
    str
        A filename stem safe for all major operating systems, e.g.
        'CCO' → 'CCO', 'c1ccccc1' → 'c1ccccc1',
        'CC(=O)O' → 'CC__O_O'.
    """
    # Replace every character that is not a letter, digit, hyphen, or
    # underscore with a single underscore.
    safe = re.sub(r"[^A-Za-z0-9\-_]", "_", smiles)

    # Collapse consecutive underscores that arise from multi-char sequences
    # such as '(=O)' → '____' → '_' for a cleaner name.
    safe = re.sub(r"_+", "_", safe)

    # Strip leading/trailing underscores.
    return safe.strip("_") or "molecule"


def visualize_molecule(
    smiles: str,
    output_dir: Path | str | None = None,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> Path:
    """Validate a SMILES string, render a 2D structure, and save it as a PNG.

    Steps performed
    ---------------
    1. Strip and reject empty input early.
    2. Parse the SMILES with RDKit — raises ValueError on failure.
    3. Canonicalize the SMILES so structurally identical molecules produce the
       same filename regardless of the input representation.
    4. Compute 2D atom coordinates with RDKit's coordinate generator.
    5. Render the structure to a PIL image and save it as PNG.
    6. Return the absolute path to the saved file.

    Parameters
    ----------
    smiles:
        Input SMILES string (e.g. 'CCO', 'c1ccccc1', 'CC(=O)O').
    output_dir:
        Directory where the PNG is saved. Defaults to
        ``<project_root>/generated_molecules/``. Created automatically if it
        does not exist.
    image_size:
        ``(width, height)`` of the rendered PNG in pixels.
        Defaults to ``(400, 400)``.

    Returns
    -------
    Path
        Absolute path to the saved PNG file.

    Raises
    ------
    ValueError
        If ``smiles`` is empty or cannot be parsed as a valid molecule.
    OSError
        If the output directory cannot be created or the file cannot be written.
    """

    # --- 1. Input validation ---------------------------------------------------

    cleaned = smiles.strip()
    if not cleaned:
        raise ValueError("Invalid SMILES: input string is empty.")

    # --- 2. RDKit parsing (chemistry validation) --------------------------------

    # Chem.MolFromSmiles returns None for any string that does not represent a
    # chemically valid molecule. We check this explicitly and raise a clear
    # error rather than letting a None propagate into the drawing code.
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        raise ValueError(
            f"Invalid SMILES: {smiles!r} could not be parsed as a valid molecule. "
            "Please check for typos or unsupported atom/bond types."
        )

    # --- 3. Canonicalization ---------------------------------------------------

    # Produce the RDKit canonical form so that e.g. 'OCC' and 'CCO' both
    # resolve to the same canonical string (and therefore the same filename).
    canonical_smiles = Chem.MolToSmiles(mol)

    # --- 4. 2D coordinate generation ------------------------------------------

    # RDKit's Compute2DCoords assigns (x, y) positions to each atom so the
    # drawing engine knows where to place them. Without this step the renderer
    # falls back to a layout that may look cluttered.
    from rdkit.Chem import AllChem  # local import keeps top-level imports light
    AllChem.Compute2DCoords(mol)

    # --- 5. Render and save ----------------------------------------------------

    # Resolve and create the output directory on demand so callers never have
    # to worry about creating it themselves.
    save_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build a descriptive filename from the canonical SMILES.
    stem = _smiles_to_filename(canonical_smiles)
    image_path = save_dir / f"{stem}.png"

    # Draw.MolToImage returns a PIL Image object. We save it directly as PNG
    # without storing it in a variable to keep memory usage minimal.
    image = Draw.MolToImage(mol, size=image_size)
    image.save(str(image_path))

    return image_path
