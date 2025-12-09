import networkx as nx
from utils.config import nlp

# small helper to build a short phrase for a token (compounds + token)
def phrase_for_token(tok):
    parts = []
    for child in tok.lefts:
        if child.dep_ in ("compound", "amod"): #, "det"):
            parts.append(child.text)
    parts.append(tok.text)
    for child in tok.rights:
        if child.dep_ in ("compound",):
            parts.append(child.text)
    return " ".join(parts).strip()


def combine_graph_elements(nodes, edges, triples):
    """
    nodes: list[str]
    edges: list[(source, target)]
    triples: list[(s,r,o)]  # or you can pass a string triple
    
    Returns: single string
    """
    node_str = ", ".join(nodes)

    edge_str = "; ".join([f"{src} -> {dst}" for (src, dst) in edges])

    # triples can be list[tuple] or already a string
    if isinstance(triples, str):
        triple_str = triples
    else:
        triple_str = "; ".join([f"{s} {r} {o}" for (s, r, o) in triples])

    # Final combined representation
    return (
        f"NODES: {node_str}. "
        f"EDGES: {edge_str}. "
        f"TRIPLES: {triple_str}."
    )

def format_triple_sci(sentence_triple):
    s, r, o = sentence_triple
    s = str(s).replace("_", " ")
    r = str(r).replace("_", " ")
    o = str(o).replace("_", " ")

    # More natural scientific phrasing
    return f"{s} {r} {o}."

def scibert_friendly_text(nodes, triples):
    """
    Convert graph structure into natural scientific prose.
    SciBERT/SPECTER encode this far better than list-format text.
    """

    # Clean node list
    node_list = ", ".join(sorted(nodes))

    # Convert triples into scientific statements
    triple_sents = []
    for s, r, o in triples:
        # Make the relation verbal and scientific
        if r in ["affects", "impact", "interact"]:
            sent = f"{s} {r}s {o}."
        elif r in ["have", "has"]:
            sent = f"{s} exhibits {o}."
        elif r == "produce":
            sent = f"{s} produces {o}."
        else:
            sent = f"{s} is related to {o}."

        triple_sents.append(sent)

    triple_block = " ".join(triple_sents)

    # Build scientific paragraph
    text = (
        f"This work discusses the following key scientific concepts: {node_list}. "
        f"The analysis reveals several relationships among them. {triple_block}"
    )

    return text


def extract_triples(t: str):
    t = " ".join(t.split("\n"))
    kg = nx.DiGraph()
    if not t:
        return kg
    doc = nlp(t)
    # add entities as nodes for clarity (if available)
    for ent in doc.ents:
        kg.add_node(ent.text, type=ent.label_)
    for sent in doc.sents:
        for tok in sent:
            if tok.pos_ == "VERB":
                subj = None
                obj = None
                for ch in tok.children:
                    if ch.dep_ in ("nsubj", "nsubjpass") and ch.pos_ in ("NOUN", "PROPN"): #, "PRON"):
                        subj = phrase_for_token(ch)
                    if ch.dep_ in ("dobj", "obj", "pobj") and ch.pos_ in ("NOUN", "PROPN"): #, "PRON"):
                        obj = phrase_for_token(ch)
                ####also allow prepositional object as object if no direct object
                if not obj:
                    for ch in tok.children:
                        if ch.dep_ == "prep":
                            for pc in ch.children:
                                if pc.dep_ == "pobj" and pc.pos_ in ("NOUN", "PROPN"):
                                    obj = phrase_for_token(pc)
                                    break
                if subj:
                    kg.add_node(subj, type="ARG")
                if obj:
                    kg.add_node(obj, type="ARG")
                if subj and obj:
                    rel = tok.lemma_.lower()
                    kg.add_edge(subj, obj, relation=rel, sentence=sent.text.strip())
    # Keep only the largest connected component to reduce noise
    # if kg.number_of_nodes() > 0:
    #     comps = list(nx.weakly_connected_components(kg))
    #     if len(comps) > 1:
    #         largest = max(comps, key=len)
    #         rm = set(kg.nodes()) - set(largest)
    #         kg.remove_nodes_from(rm)
    # print(kg.nodes)
    # print(kg.edges)

    edge_attrs = [(u, d.get("relation"), v, d.get("sentence")) for u, v, d in kg.edges(data=True)]
    
    ####V1
    # simple_triples = [(u, rel, v) for u, rel, v, _ in edge_attrs]
    # string = combine_graph_elements(kg.nodes, kg.edges, " ; ".join([f"{s} {r} {o}" for (s, r, o) in simple_triples]))
    # print(string)

    # formatted_triples = [format_triple_sci(t) for t in simple_triples]
    # print("formated triples: ", formatted_triples)

    # triple_block = " ".join(formatted_triples)
    # print("triple block:" , triple_block)

    # string = combine_graph_elements(
    #     kg.nodes,
    #     kg.edges,
    #     triple_block
    # )
    # print(string)

    ###V2
    simple_triples = [(u, rel, v) for u, rel, v, _ in edge_attrs]
    string = scibert_friendly_text(kg.nodes, simple_triples)
    # print(string)

    return string