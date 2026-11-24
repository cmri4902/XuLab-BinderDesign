# `CD19_ECD.pdb` — human CD19 extracellular domain target record

This is the provenance and limitation record for the target file the
`cd19_scfv_discovery` campaign designs against. Every number here was read out of
the deposited coordinates, not recalled or assumed; the commands that reproduce
them are at the end.

## Source

| Item | Value |
| --- | --- |
| RCSB entry | **6AL5** |
| Title | COMPLEX BETWEEN CD19 (N138Q MUTANT) AND B43 FAB |
| Method / resolution | X-ray diffraction, **3.0 Å** |
| Primary citation | Teplyakov A, Obmolova G, Gilliland GL. *Crystal structure of B-cell co-receptor CD19 in complex with antibody B43 reveals an unexpected fold.* Proteins, 2018 |
| CD19 chain used | **chain A** (entity 1, "B-lymphocyte antigen CD19", *Homo sapiens*) |
| Chains discarded | H and L (B43 Fab heavy and light chain) |
| Heteroatoms discarded | 2 NAG (N-glycans), 1 PCA (Fab N-terminal pyroglutamate), waters |
| Numbering | deposited author numbering = **UniProt P15391** full-length numbering (signal peptide counted); not renumbered |

Why 6AL5 rather than 7JIC: 7JIC (cryo-EM, 3.8 Å, *Science* 2021) is the
CD19–CD81 co-receptor complex with coltuximab Fab, and its CD19 chain carries the
transmembrane helix plus a FLAG/TEV expression tag. 6AL5 is an X-ray structure of
the ectodomain alone at higher resolution with only antibody chains to strip, so
it is the cleaner target file. 7JIC is still used below, as independent evidence
about the epitope and about CD81 occlusion.

## What the prepared file contains

* Chain **A**, residues **21–277**, 227 residues, 1656 atoms.
* Ordered segments: **21–39, 48–137, 153–176, 184–277**.
* Absent from the deposited model, therefore absent here: **40–47, 138–152,
  177–183**. No coordinate was modelled in to fill them.
* Sequence as present (gaps simply concatenated — this is not a continuous
  construct sequence):

  ```
  EEPLVVKVEEGDNAVLQCLQQLTWSRESPLKPFLKLSLGLPGLGIHMRPLAIWLFIFNVSQQMGGFYLCQPGPP
  SEKAWQPGWTVNVEGSGELFRWNVSDLGGLGCGLKSPKLYVWAKDRPEIWEGEPPCLPPLSQDLTMAPGSTLWL
  SCGVPPDSVSRGPLSWTHVHPKGPKSLLSLELKDDRPARDMWVMETGLLLPRATAQDAGKYYCHRGNLTMSFHL
  EITAR
  ```

### Construct differences, and why none of them is in the file

* **N138Q engineered mutation.** The deposited construct replaces Asn138 with
  Gln to remove an N-glycosylation sequon (`SEQADV 6AL5 GLN A 138 UNP P15391 ASN
  138 ENGINEERED MUTATION`). Residue 138 sits inside the unplaced 138–152
  segment, so the mutated residue is **not present in the coordinates at all**.
  The prepared file therefore carries no engineered mutation — but it also
  carries no 138–152 loop, which is the more important limitation (below).
* **C-terminal GSHHHHHH expression tag**, modelled from residue 278 onward
  (`SEQADV` lines 278–285). Cut: the file stops at the last native residue, 277.
* CD19 is a glycoprotein. The two modelled N-glycans (on **Asn86** and
  **Asn125**) were removed with the other heteroatoms, so the design runs against
  the unglycosylated protein surface.

## Epitope chosen, and why it is plausible on a B cell

The hotspots were not chosen by eye. They are residues that a real antibody Fab
actually contacts in an experimental structure, and they were cross-checked
against a second, independent structure solved with a different antibody.

**B43 epitope in 6AL5** — CD19 chain A residues with a heavy atom within 4.0 Å of
chain H or L (contact-atom count in brackets):

L96 (1), S103 (3), E120 (1), K155 (8), Y157 (5), W159 (2), D162 (9), R163 (4),
P164 (1), I166 (5), V217 (2), H218 (2), P219 (8), **K220 (32)**, G221 (3),
P222 (7), K223 (8), S224 (11)

**Coltuximab epitope in 7JIC** — CD19 chain A residues within 4.0 Å of chains H
or L: G67, Q90, Y157, D162, R163, P164, E165, H218, P219, K220, G221, P222, K223,
S224.

**Consensus of the two** (contacted by both antibodies): **Y157, D162, R163,
P164, H218, P219, K220, G221, P222, K223, S224**.

### Hotspots used by the campaign

```
A162,A219,A220,A222
```

Each of the four is in the two-antibody consensus, each has ≥ 7 contact atoms in
6AL5, and the four form one compact patch (all Cα–Cα distances ≤ 14 Å, most 4–11
Å), so they describe a single site rather than a scattered selection. They are a
*subset* of the epitope, used to steer where the binder is placed; the full
consensus above is the site itself.

### Accessibility checks on that site

* **Membrane.** The hotspot Cα atoms are **47–53 Å** from residue 277, the last
  modelled residue before the membrane-proximal stalk and transmembrane helix.
  The site is at the membrane-distal end of the ectodomain.
* **CD81.** In 7JIC, CD19 contacts CD81 at N33, V35, Q37, W81, F83, F85, W111,
  D187, L188, T189, M190, G193, S194, T195, W197, P203, W240, M242, L247, T292,
  Y295, L296, C299, L300, L303, R311. **None of these is in the epitope above**,
  so the chosen site is not part of the CD81 interface. Coltuximab in fact binds
  CD19 while CD81 is bound, in that same structure.
* **Glycans.** The nearest modelled glycosylation site, **Asn125**, is 9–19 Å
  from the hotspot Cα atoms, and **Asn86** is 18–21 Å away. Neither is in the
  epitope, but a full glycan is large, so partial shielding of the site on native
  CD19 cannot be ruled out from these coordinates.

### Limitations to carry into any interpretation

1. **The 138–152 loop is missing**, and in native CD19 it carries an
   N-glycosylation site at **Asn138** that this construct deliberately removed.
   Its anchor residues (137 and 153) are 13–28 Å from the hotspots, so the native
   loop and its glycan could lie near the chosen site. A design predicted here
   has never seen that loop. This is the single largest caveat on the epitope.
2. **3.0 Å X-ray coordinates** of a crystallised, deglycosylation-mutant,
   tag-bearing, detergent-free ectodomain fragment. Side-chain positions at the
   interface are modelled at that resolution.
3. The two other unplaced segments (40–47, 177–183) are also absent, so the
   target surface presented to the designer is incomplete in three places.
4. CD19 on a B cell sits in a complex with CD81 and in a signalling assembly with
   the B-cell receptor. Nothing about steric access in that assembly, about
   epitope availability on a tumour cell, or about internalisation can be read
   from this file.

## Reproducing this file

This target is the platform's **default example**, not a special case: it was
built with the same general-purpose tool any other target goes through.

```bash
# look first - this refuses to choose a chain in a three-chain complex
python lab/scripts/prepare_target.py --entry 6AL5 --describe

# then prepare it
python lab/scripts/prepare_target.py --entry 6AL5 --chain A \
    --first 21 --last 277 --name hCD19_ECD \
    --output lab/targets/CD19_ECD.pdb --contacts
```

`--entry` needs internet access, which the HPC compute nodes do not have; that is
why the prepared file is committed to the repository. With the entry already
downloaded, `--source 6AL5.pdb` replaces `--entry 6AL5` and needs no network.

The `--contacts` output is the B43 epitope table quoted above. The 7JIC numbers
were read from `https://files.rcsb.org/download/7JIC.cif` with the same 4.0 Å
heavy-atom criterion.

The file's own `REMARK 999` header carries the same provenance: the entry, the
method and resolution, the entity, the selection, the residues present, the
ordered and absent segments, everything removed, each `SEQADV` the source
declares, and the SIFTS mapping from the entity's numbering to UniProt P15391.
`python lab/scripts/prepare_target.py --source lab/targets/CD19_ECD.pdb
--describe` reads it back.
