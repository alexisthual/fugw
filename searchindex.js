Search.setIndex({"docnames": ["auto_examples/00_basics/index", "auto_examples/00_basics/plot_1_dense", "auto_examples/00_basics/plot_2_1_lmds", "auto_examples/00_basics/plot_2_2_coarse_to_fine", "auto_examples/00_basics/sg_execution_times", "auto_examples/01_brain_alignment/index", "auto_examples/01_brain_alignment/plot_1_aligning_brain_dense", "auto_examples/01_brain_alignment/plot_2_aligning_brain_sparse", "auto_examples/01_brain_alignment/sg_execution_times", "auto_examples/index", "index", "modules/mappings", "modules/mappings.FUGW", "modules/mappings.FUGWSparse", "modules/solvers", "modules/solvers.FUGWSolver", "modules/solvers.FUGWSparseSolver", "pages/api_references", "pages/contributing", "pages/introduction"], "filenames": ["auto_examples/00_basics/index.rst", "auto_examples/00_basics/plot_1_dense.rst", "auto_examples/00_basics/plot_2_1_lmds.rst", "auto_examples/00_basics/plot_2_2_coarse_to_fine.rst", "auto_examples/00_basics/sg_execution_times.rst", "auto_examples/01_brain_alignment/index.rst", "auto_examples/01_brain_alignment/plot_1_aligning_brain_dense.rst", "auto_examples/01_brain_alignment/plot_2_aligning_brain_sparse.rst", "auto_examples/01_brain_alignment/sg_execution_times.rst", "auto_examples/index.rst", "index.rst", "modules/mappings.rst", "modules/mappings.FUGW.rst", "modules/mappings.FUGWSparse.rst", "modules/solvers.rst", "modules/solvers.FUGWSolver.rst", "modules/solvers.FUGWSparseSolver.rst", "pages/api_references.rst", "pages/contributing.md", "pages/introduction.md"], "titles": ["Basics", "Transport distributions using dense solvers", "Generate embeddings from mesh", "Transport distributions using sparse solvers", "Computation times", "Brain alignment", "Align brain surfaces of 2 individuals with fMRI data", "Align high-resolution brain surfaces of 2 individuals with fMRI data", "Computation times", "Examples", "Fused Unbalanced Gromov-Wasserstein for Python", "<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fugw.mappings</span></code>", "<code class=\"docutils literal notranslate\"><span class=\"pre\">FUGW</span></code>", "<code class=\"docutils literal notranslate\"><span class=\"pre\">FUGWSparse</span></code>", "<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">fugw.solvers</span></code>", "<code class=\"docutils literal notranslate\"><span class=\"pre\">FUGWSolver</span></code>", "<code class=\"docutils literal notranslate\"><span class=\"pre\">FUGWSparseSolver</span></code>", "API references", "Contributing", "Introduction"], "terms": {"transport": [0, 4, 6, 7, 9, 10, 12, 13, 15, 16, 19], "distribut": [0, 2, 4, 7, 9, 12, 13, 15, 16, 19], "us": [0, 2, 4, 9, 10, 11, 12, 13, 15, 16, 17, 19], "dens": [0, 3, 4, 6, 7, 9, 12, 15], "solver": [0, 4, 6, 7, 9, 10, 12, 13, 15, 16], "gener": [0, 1, 3, 4, 6, 7, 9], "embed": [0, 1, 3, 4, 7, 9, 13], "from": [0, 1, 3, 4, 6, 7, 9, 12, 13, 19], "mesh": [0, 4, 6, 7, 9, 12, 13], "spars": [0, 1, 4, 7, 9, 13, 16], "click": [1, 2, 3, 6, 7], "here": [1, 2, 3, 6, 7], "download": [1, 2, 3, 6, 7], "full": [1, 2, 3, 6, 7], "exampl": [1, 2, 3, 6, 7], "code": [1, 2, 3, 6, 7, 19], "In": [1, 2, 3, 6, 7, 10, 12, 13, 19], "thi": [1, 2, 3, 6, 7, 11, 12, 13, 15, 16, 17, 19], "we": [1, 2, 3, 6, 7, 15, 16, 19], "sampl": [1, 2, 3, 7, 15, 16], "2": [1, 2, 3, 5, 8, 9, 11, 12, 13, 15, 16, 17, 19], "toi": [1, 3], "fugw": [1, 2, 3, 6, 7, 10, 13, 15, 16, 19], "align": [1, 2, 3, 8, 10, 19], "between": [1, 2, 3, 6, 7, 12, 13, 15, 16, 19], "them": [1, 3, 6, 7, 10, 11, 17, 19], "ar": [1, 3, 6, 7, 12, 13, 19], "typic": [1, 3, 7], "when": [1, 2, 3], "both": [1, 3], "have": [1, 2, 3, 6, 7, 19], "less": [1, 7], "than": [1, 3, 7], "10k": [1, 3, 7], "point": [1, 2, 3, 12, 13, 15, 16, 19], "import": [1, 2, 3, 6, 7, 12, 13], "matplotlib": [1, 2, 3, 6, 7], "pyplot": [1, 2, 3, 6, 7], "plt": [1, 2, 3, 6, 7], "numpi": [1, 2, 3, 6, 7], "np": [1, 2, 3, 6, 7], "torch": [1, 2, 3, 6, 7, 12, 13, 15, 16], "util": [1, 3, 6, 7, 12, 13], "init_mock_distribut": [1, 3], "collect": [1, 3, 6, 7], "linecollect": [1, 3], "random": [1, 2, 3, 7], "seed": [1, 3, 7], "manual_se": [1, 2, 3, 7], "n_points_sourc": [1, 3], "50": [1, 3], "n_points_target": [1, 3], "40": [1, 6, 7], "n_features_train": [1, 3], "n_features_test": [1, 3], "let": [1, 2, 3, 6, 7], "u": [1, 2, 3, 19], "train": [1, 3], "data": [1, 2, 3, 5, 8, 9, 11, 12, 13, 17], "sourc": [1, 2, 3, 6, 7, 12, 13, 15, 16], "target": [1, 3, 6, 7, 12, 13, 15, 16], "_": [1, 3, 6], "source_features_train": [1, 3], "source_geometri": [1, 6, 12], "source_embed": [1, 3], "target_features_train": [1, 3], "target_geometri": [1, 6, 12], "target_embed": [1, 3], "shape": [1, 2, 3, 6, 7], "size": [1, 2, 3, 6, 7, 15, 16], "can": [1, 2, 3, 6, 7, 10, 12, 13, 19], "visual": [1, 3, 7], "featur": [1, 3, 12, 13, 15, 16, 19], "fig": [1, 2, 3, 6, 7], "figur": [1, 2, 3, 6, 7, 19], "figsiz": [1, 2, 3, 6, 7], "4": [1, 3, 6, 7, 10, 19], "ax": [1, 2, 3, 6, 7], "add_subplot": [1, 2, 3, 6, 7], "set_titl": [1, 2, 3, 6, 7], "set_aspect": [1, 3], "equal": [1, 3, 19], "datalim": [1, 3], "scatter": [1, 2, 3], "0": [1, 2, 3, 6, 7, 10, 12, 13, 15, 16], "1": [1, 2, 3, 6, 7, 8, 12, 13, 15, 16, 19], "label": [1, 3, 6, 7], "legend": [1, 3, 6, 7], "show": [1, 2, 3, 6, 7], "And": [1, 3, 6], "project": [1, 2, 6, 7], "3d": [1, 2, 6, 7], "ie": [1, 19], "geometri": [1, 12, 13, 15, 16, 19], "": [1, 2, 3, 6, 7, 12, 13, 19], "15": [1, 2, 6, 7], "should": [1, 2, 3, 6, 7, 10, 12, 13, 19], "normal": [1, 3, 12, 13], "befor": [1, 3, 6, 7, 15, 16], "call": [1, 14, 17, 19], "source_features_train_norm": [1, 3], "linalg": [1, 3, 6, 7], "norm": [1, 3, 6, 7, 13], "dim": [1, 3], "reshap": [1, 3, 6, 7], "target_features_train_norm": [1, 3], "source_geometry_norm": [1, 6], "max": [1, 6, 7], "target_geometry_norm": [1, 6], "defin": [1, 3], "optim": [1, 3, 6, 10], "problem": [1, 3, 10, 12, 13], "solv": [1, 3, 12, 13, 14, 15, 16], "alpha": [1, 3, 6, 7, 12, 13, 15, 16], "5": [1, 2, 3, 6, 7, 12, 13, 15, 16, 19], "ep": [1, 3, 6, 7, 12, 13, 15, 16], "1e": [1, 3, 6, 7, 12, 13, 15, 16], "now": [1, 2, 3, 6, 19], "fit": [1, 2, 3, 6, 7, 11, 12, 13, 15, 16, 17], "plan": [1, 3, 6, 7, 12, 13, 15, 16, 19], "sinkhorn": [1, 6, 12, 15, 16, 19], "verbos": [1, 2, 3, 6, 7, 12, 13, 15, 16], "true": [1, 2, 3, 6, 7], "16": [1, 3, 6, 7], "38": [1, 3, 7], "06": [1, 8, 15, 16], "bcd": [1, 3, 6, 7, 15, 16, 19], "step": [1, 3, 6, 7, 15, 16], "10": [1, 2, 3, 6, 7, 8, 10, 15, 16, 19], "loss": [1, 3, 6, 7, 12, 13, 15, 16, 19], "07106098532676697": 1, "py": [1, 2, 3, 4, 6, 7, 8], "360": [1, 3, 6, 7], "base": [1, 3, 6, 7, 19], "0716898962855339": 1, "entrop": [1, 3, 6, 7, 15, 16], "07": [1, 15, 16], "06062595546245575": 1, "06126530095934868": 1, "3": [1, 2, 3, 4, 6, 7, 19], "05847756192088127": 1, "05912858247756958": 1, "0584559366106987": 1, "059106577187776566": 1, "08": [1, 4, 6], "05841211974620819": 1, "059064075350761414": 1, "6": [1, 3, 19], "0584060437977314": 1, "059058528393507004": 1, "09": [1, 3, 6, 7], "7": [1, 3, 19], "058397069573402405": 1, "05904938280582428": 1, "8": [1, 2, 3, 4, 6, 7], "058390866965055466": 1, "05904326215386391": 1, "9": [1, 2, 3, 4], "05838455259799957": 1, "05903756245970726": 1, "05838431790471077": 1, "059037402272224426": 1, "The": [1, 2, 3, 6, 19], "access": [1, 3, 7], "after": [1, 3], "model": [1, 3, 6, 7, 11, 17], "ha": [1, 6, 7], "been": [1, 3], "pi": [1, 3, 6, 7, 15, 16], "print": [1, 3, 6, 7], "f": [1, 3, 6, 7, 15, 16], "total": [1, 2, 3, 4, 6, 7, 8], "mass": [1, 3, 12, 13, 19], "sum": [1, 3, 12, 13], "5f": [1, 3], "99099": 1, "i": [1, 2, 3, 6, 7, 10, 11, 12, 13, 15, 16, 17, 19], "evolut": [1, 3, 6, 7], "dure": [1, 3, 6, 7, 15, 16], "without": [1, 3, 6, 7], "term": [1, 3, 6, 7, 12, 13], "subplot": [1, 3, 6, 7], "set_ylabel": [1, 3, 6, 7], "set_xlabel": [1, 3, 6, 7], "plot": [1, 3, 6, 7], "loss_step": [1, 3, 6, 7, 15, 16], "loss_entrop": [1, 3, 6, 7, 15, 16], "store": [1, 3, 6, 7, 15, 16], "tensor": [1, 3, 6, 15, 16], "small": [1, 6], "enough": [1, 6, 7], "displai": [1, 6, 7], "altogeth": [1, 6], "vertic": [1, 2, 3, 6, 7, 13, 19], "im": [1, 2, 6, 7], "imshow": [1, 2, 6, 7], "cmap": [1, 6, 7], "viridi": [1, 6, 7], "colorbar": [1, 2, 6, 7], "shrink": [1, 2, 6, 7], "previou": [1, 19], "tell": 1, "veri": [1, 3, 6, 19], "anoth": [1, 19], "inform": [1, 6, 7, 19], "wai": 1, "look": [1, 2, 3, 6, 7, 19], "consist": 1, "check": [1, 2], "which": [1, 2, 3, 6, 7, 11, 15, 16, 17, 19], "were": [1, 6, 7, 19], "match": [1, 3, 6, 7, 19], "togeth": 1, "space": [1, 3, 12, 13], "ndisplai": [1, 3], "draw": [1, 3], "line": [1, 3, 6, 7], "indic": [1, 3, 6, 7], "cartesian_prod": [1, 3], "arang": [1, 3], "segment": [1, 3], "stack": [1, 3, 6, 7], "permut": [1, 3], "pi_norm": [1, 3], "line_seg": [1, 3], "flatten": [1, 3], "color": [1, 3, 6, 7], "black": [1, 3], "lw": [1, 3], "zorder": [1, 3], "add_collect": [1, 3], "final": [1, 2, 3, 6, 7, 19], "unseen": [1, 3, 6, 7], "source_features_test": [1, 3], "rand": [1, 3], "target_features_test": [1, 3], "transformed_data": [1, 3], "transform": [1, 3, 6, 7, 11, 12, 13, 17], "assert": [1, 2, 3], "run": [1, 2, 3, 6, 7, 10, 12, 13, 15, 16, 19], "time": [1, 2, 3, 6, 7], "script": [1, 2, 3, 6, 7], "minut": [1, 2, 3, 6, 7], "031": [1, 4], "second": [1, 2, 3, 6, 7, 19], "estim": [1, 2, 3, 6], "memori": [1, 2, 3, 6, 7], "usag": [1, 2, 3, 6, 7], "270": 1, "mb": [1, 2, 3, 4, 6, 7, 8], "python": [1, 2, 3, 6, 7], "plot_1_dens": [1, 4], "jupyt": [1, 2, 3, 6, 7], "notebook": [1, 2, 3, 6, 7], "ipynb": [1, 2, 3, 6, 7], "galleri": [1, 2, 3, 6, 7, 9], "sphinx": [1, 2, 3, 6, 7, 9], "how": [2, 6, 7, 19], "deriv": [2, 7, 19], "an": [2, 6, 7, 10, 15, 16, 19], "approxim": [2, 6, 7, 13, 19], "kernel": [2, 6, 12, 15, 19], "matrix": [2, 6, 7, 12, 13, 15, 16, 19], "geodes": [2, 6, 7], "distanc": [2, 6, 7, 12, 13, 19], "given": [2, 12, 13], "techniqu": 2, "try": [2, 3, 19], "larg": 2, "number": [2, 3, 6, 7, 12, 13, 15, 16, 19], "inde": [2, 6, 7], "pairwis": 2, "won": [2, 7, 10], "t": [2, 3, 7, 10, 19], "comput": [2, 3, 12, 13, 15, 16, 19], "right": [2, 6, 7], "dimens": [2, 7], "probabl": [2, 6, 7], "gdist": [2, 6], "lmd": [2, 7], "nilearn": [2, 6, 7], "dataset": [2, 6, 7], "surfac": [2, 5, 8, 9], "exact": [2, 19], "each": [2, 3, 6, 7, 11, 12, 13, 15, 16, 17, 19], "vertex": [2, 6, 7], "n_landmark": [2, 7], "k": [2, 7, 13, 16, 19], "100": [2, 3, 7], "load": [2, 6, 7], "pre": [2, 10], "first": [2, 6, 7, 12, 13, 19], "fsaverage3": [2, 6], "fetch_surf_fsaverag": [2, 6, 7], "coordin": [2, 6, 7, 15, 16, 19], "triangl": [2, 6, 7], "load_surf_mesh": [2, 6, 7], "pial_left": [2, 6, 7], "creat": [2, 6, 7], "github": [2, 3, 6, 10], "home": [2, 6], "nilearn_data": [2, 6], "http": [2, 3, 6, 10, 19], "osf": [2, 6], "io": [2, 6], "asvjk": 2, "done": [2, 6], "min": [2, 6], "extract": 2, "b9ce491b47822c5b4950eeeb75d15a92": 2, "642": [2, 6], "plot_trisurf": 2, "easi": [2, 7], "parallel": [2, 7], "x": [2, 3, 7, 13, 15, 16], "compute_lmd": [2, 7], "n_job": [2, 7], "geodesic_dist": [2, 7], "landmark": [2, 7, 19], "00": [2, 3, 4, 7], "02": 2, "It": [2, 7, 11, 17], "correct": 2, "actual": [2, 6, 7], "peek": 2, "pair": [2, 3, 7, 15, 16], "well": [2, 6, 19], "211": 2, "true_kernel_matrix": 2, "local_gdist_matrix": [2, 6], "astyp": [2, 6], "float64": [2, 6], "int32": [2, 6], "toarrai": [2, 6], "212": 2, "approximated_kernel_matrix": 2, "cdist": 2, "975": [2, 4], "138": [2, 4], "plot_2_1_lmd": [2, 4], "more": [3, 6, 7], "map": [3, 12, 13, 14], "fugwspars": [3, 7, 11, 17], "coarse_to_fin": [3, 7], "scipi": [3, 7], "coo_matrix": [3, 7], "300": 3, "n_samples_sourc": 3, "n_samples_target": 3, "do": 3, "forget": 3, "source_embeddings_norm": [3, 7], "source_d_max": 3, "random_norm": [3, 7], "target_embeddings_norm": [3, 7], "target_d_max": 3, "coars": [3, 7], "fine": [3, 7], "grain": [3, 6, 7], "also": [3, 6, 7, 10, 19], "specifi": 3, "coarse_map": [3, 7], "coarse_mapping_solv": [3, 7], "mm": [3, 6, 7, 12, 13, 15, 16, 19], "coarse_mapping_solver_param": [3, 7], "tol_uot": [3, 6, 7, 15, 16], "fine_map": [3, 7], "fine_mapping_solv": [3, 7], "fine_mapping_solver_param": [3, 7], "limit": [3, 6], "you": [3, 6, 7, 10, 11, 12, 13, 17], "carefulli": 3, "set": [3, 6, 7, 12, 13], "selection_radiu": 3, "thei": [3, 6, 7, 19], "greatli": 3, "affect": [3, 7], "sparsiti": [3, 7, 13], "usal": 3, "domain": 3, "knowledg": 3, "relat": 3, "source_featur": [3, 6, 7, 12, 13], "target_featur": [3, 6, 7, 12, 13], "source_geometry_embed": [3, 7, 13], "target_geometry_embed": [3, 7, 13], "parametr": [3, 7], "source_sample_s": [3, 7], "target_sample_s": [3, 7], "select": [3, 7], "present": [3, 7, 19], "mask": [3, 6, 7, 13], "coarse_pairs_selection_method": [3, 7], "topk": [3, 7], "source_selection_radiu": [3, 7], "target_selection_radiu": [3, 7], "misc": [3, 7], "22": [3, 19], "028055913746356964": 3, "02814635820686817": 3, "025683855637907982": 3, "025863073766231537": 3, "23": 3, "016268832609057426": 3, "01665913127362728": 3, "011751125566661358": 3, "012296381406486034": 3, "010866605676710606": 3, "011479916982352734": 3, "24": [3, 6], "010522968135774136": 3, "011175187304615974": 3, "010338868945837021": 3, "011017151176929474": 3, "25": [3, 7, 8], "010221882723271847": 3, "010919460095465183": 3, "26": [3, 7, 19], "0101405568420887": 3, "010853287763893604": 3, "27": [3, 6], "0100807324051857": 3, "01080571860074997": 3, "200": [3, 7], "01": [3, 4, 7, 12, 13, 15, 16], "workspac": 3, "src": 3, "51": 3, "userwarn": [3, 6, 7], "csr": [3, 13], "support": 3, "beta": 3, "state": 3, "If": [3, 10, 12, 13], "miss": 3, "function": [3, 6, 7, 15, 16, 19], "pleas": [3, 10], "submit": 3, "request": 3, "com": [3, 10, 19], "pytorch": [3, 10], "issu": [3, 19], "trigger": 3, "intern": [3, 10], "aten": 3, "sparsecsrtensorimpl": 3, "cpp": 3, "54": [3, 6], "return": [3, 6, 7, 12, 13, 15, 16], "devic": [3, 12, 13], "dtype": 3, "to_sparse_csr": 3, "39": [3, 7], "021763041615486145": 3, "478": [3, 7], "02189093828201294": 3, "42": [3, 6], "017754876986145973": 3, "01801164820790291": 3, "45": [3, 7], "011553710326552391": 3, "012020763009786606": 3, "47": 3, "009638299234211445": 3, "010233832523226738": 3, "008976257406175137": 3, "009645415470004082": 3, "52": 3, "008657296188175678": 3, "009374301880598068": 3, "55": [3, 4], "008474555797874928": 3, "009225510992109776": 3, "59": 3, "008358475752174854": 3, "009134968742728233": 3, "04": [3, 7], "008279440924525261": 3, "009075913578271866": 3, "008222405798733234": 3, "009034985676407814": 3, "99800": 3, "99842": 3, "about": [3, 6, 7, 19], "60": 3, "what": [3, 6, 7], "equival": [3, 7, 19], "would": [3, 6, 7], "high": [3, 5, 6, 8, 9, 12, 13], "usual": [3, 6, 7], "want": [3, 6, 7, 11, 17], "keep": [3, 6, 7], "much": [3, 6, 7], "smaller": 3, "sparsity_ratio": 3, "valu": [3, 6, 7, 12, 13, 15, 16, 19], "numel": [3, 7], "ratio": 3, "non": [3, 6, 7, 19], "null": [3, 7], "coeffici": [3, 7], "2f": [3, 6, 7], "63": [3, 6, 7], "17": [3, 6], "particular": 3, "don": 3, "expect": [3, 7], "particularli": 3, "meaning": [3, 7], "structur": [3, 7, 19], "fine_mapping_as_scipy_coo": [3, 7], "spy": [3, 7], "precis": [3, 7], "markers": [3, 7], "observ": 3, "to_dens": 3, "815": [3, 4], "1544": 3, "plot_2_2_coarse_to_fin": [3, 4], "12": [4, 7], "820": 4, "execut": [4, 8], "auto_examples_00_bas": 4, "file": [4, 8], "1543": 4, "269": 4, "individu": [5, 8, 9, 10, 12, 13, 19], "fmri": [5, 8, 9, 12, 13], "resolut": [5, 6, 8, 9], "low": [6, 7, 12, 13], "left": [6, 7], "hemispher": [6, 7], "z": [6, 7], "score": [6, 7], "contrast": [6, 7, 12, 13], "mpl": [6, 7], "gridspec": [6, 7], "mpl_toolkit": [6, 7], "axes_grid1": [6, 7], "make_axes_locat": [6, 7], "imag": [6, 7], "volumetr": [6, 7], "per": [6, 7], "api": [6, 7, 10], "subject": [6, 7, 12, 13], "out": [6, 7], "assess": [6, 7], "qualiti": [6, 7], "our": [6, 7, 12, 13], "n_subject": [6, 7], "sentenc": [6, 7], "read": [6, 7], "v": [6, 7], "checkerboard": [6, 7], "listen": [6, 7], "calcul": [6, 7], "button": [6, 7], "press": [6, 7, 19], "n_training_contrast": [6, 7], "brain_data": [6, 7], "fetch_localizer_contrast": [6, 7], "get_anat": [6, 7], "source_imgs_path": [6, 7], "len": [6, 7], "target_imgs_path": [6, 7], "brainomics_loc": 6, "hwbm2": 6, "5d27cd441c5b4a001aa08008": 6, "5d27c03e45253a001c3e189f": 6, "5d27bfd0114a420016057cba": 6, "5d27cb281c5b4a001aa07e29": 6, "5d27cc0845253a001c3e22bd": 6, "5d27d10b114a420019044ed8": 6, "5d27d89d1c5b4a001d9f5e6": 6, "5d27d429a26b340017083380": 6, "5d27ddc91c5b4a001b9ef9d0": 6, "5d27d14f114a420019044efc": 6, "5d275eb845253a001c3dbf76": 6, "5d275ede1c5b4a001aa00c26": 6, "5d27037f45253a001c3d4563": 6, "5d7b8948fcbf44001c44e695": 6, "usr": [6, 7], "local": [6, 7, 15, 16], "lib": [6, 7], "python3": [6, 7], "site": [6, 7], "packag": [6, 7, 10], "func": [6, 7], "763": [6, 7], "legacy_format": [6, 7], "default": [6, 7, 12, 13, 15, 16], "fals": [6, 7, 12, 13, 15, 16], "releas": [6, 7], "11": [6, 7], "fetcher": [6, 7], "panda": [6, 7], "datafram": [6, 7], "instead": [6, 7, 19], "recarrai": [6, 7], "warn": [6, 7], "_legacy_format_msg": [6, 7], "like": [6, 7, 11, 12, 13, 17], "follow": [6, 7, 15, 16], "interact": [6, 7], "contrast_index": [6, 7], "view_img": [6, 7], "anat": [6, 7], "titl": [6, 7, 10], "opac": [6, 7], "_util": [6, 7], "niimg": [6, 7], "finit": [6, 7], "detect": [6, 7], "These": [6, 7], "replac": [6, 7], "zero": [6, 7], "core": [6, 7], "fromnumer": [6, 7], "784": [6, 7], "partit": [6, 7], "ignor": [6, 7], "maskedarrai": [6, 7], "kth": [6, 7], "axi": [6, 7], "kind": [6, 7], "order": [6, 7, 10], "repres": [6, 19], "cortic": [6, 7, 12, 13, 19], "aggreg": [6, 7], "build": [6, 7], "For": [6, 7, 19], "sake": [6, 7], "phase": [6, 7], "short": [6, 7, 19], "even": [6, 7], "cpu": [6, 7, 10, 12, 13], "made": [6, 7], "def": [6, 7], "load_images_and_project_to_surfac": [6, 7], "image_path": [6, 7], "load_img": [6, 7], "img": [6, 7], "surface_imag": [6, 7], "nan_to_num": [6, 7], "vol_to_surf": [6, 7], "465": [6, 7], "runtimewarn": [6, 7], "mean": [6, 7], "empti": [6, 7], "slice": [6, 7], "textur": [6, 7], "nanmean": [6, 7], "all_sampl": [6, 7], "plot_surface_map": [6, 7], "surface_map": [6, 7], "coolwarm": [6, 7], "kwarg": [6, 7], "plot_surf": [6, 7], "bg_map": [6, 7], "sulc_left": [6, 7], "bg_on_data": [6, 7], "dark": [6, 7], "grid_spec": [6, 7], "all": [6, 7, 19], "contrast_nam": [6, 7], "enumer": [6, 7], "j": [6, 7, 13, 19], "vmax": [6, 7], "vmin": [6, 7], "add": [6, 7], "rang": [6, 7], "off": [6, 7], "text": [6, 7, 19], "sub": [6, 7], "va": [6, 7], "center": [6, 7], "divid": [6, 7], "cax": [6, 7], "append_ax": [6, 7], "add_ax": [6, 7], "cm": [6, 7], "scalarmapp": [6, 7], "note": [6, 7, 19], "same": [6, 7, 12, 13], "doe": [6, 7], "case": [6, 7, 12, 13, 19], "compute_geometry_from_mesh": 6, "mesh_path": 6, "fsaverage3_pial_left_geometri": 6, "vertex_index": [6, 7], "matric": [6, 19], "contain": [6, 7, 15, 16], "anatom": [6, 7, 12, 13], "millimet": [6, 7], "other": [6, 7], "111": [6, 7], "magma": [6, 7], "cbar_tick_format": [6, 7], "scale": [6, 7, 19], "unclear": 6, "whether": [6, 7], "compar": [6, 7], "moreov": [6, 7, 19], "hyper": [6, 7], "paramet": [6, 7, 12, 13, 15, 16], "depend": [6, 10], "respect": [6, 7, 19], "empirac": 6, "lead": [6, 7], "nan": 6, "source_features_norm": [6, 7], "target_features_norm": [6, 7], "interest": 6, "similar": [6, 19], "preserv": [6, 19], "leav": [6, 19], "rho": [6, 7, 12, 13], "its": 6, "too": [6, 7, 19], "faster": [6, 7], "solwer": 6, "finer": 6, "meant": 6, "gpu": [6, 7, 10, 12, 13], "100x": 6, "slower": 6, "rememb": [6, 7], "onli": [6, 7, 19], "block": [6, 15, 16, 19], "descent": [6, 15, 16, 19], "iter": [6, 7, 15, 16], "solver_param": [6, 12, 13], "nits_bcd": [6, 7, 15, 16], "13": 6, "028870292007923126": 6, "02978154458105564": 6, "41": 6, "004489848855882883": 6, "005622141994535923": 6, "004457559436559677": 6, "005594924092292786": 6, "n": [6, 7, 12, 13, 15, 16, 19], "loss_tim": 6, "1f": [6, 7], "becaus": [6, 7], "known": 6, "commun": 6, "librari": 6, "come": [6, 11, 17], "most": [6, 7, 11, 17], "retrain": 6, "implement": [6, 10, 14, 17, 19], "maxim": [6, 7], "minim": [6, 19], "approach": [6, 19], "solut": [6, 7, 15, 16, 19], "mm_map": 6, "tol_bcd": [6, 15, 16], "03685737028717995": 6, "036940909922122955": 6, "014586355537176132": 6, "015035441145300865": 6, "21": 6, "006973237730562687": 6, "007662989664822817": 6, "005839386489242315": 6, "006633028853684664": 6, "36": 6, "005359851289540529": 6, "006219192408025265": 6, "ibpp": [6, 12, 13, 15, 16, 19], "ibpp_map": 6, "034325726330280304": 6, "03454320505261421": 6, "46": 6, "007011039648205042": 6, "00773261021822691": 6, "005309466738253832": 6, "006187541410326958": 6, "0049124350771307945": 6, "005864681676030159": 6, "004740338306874037": 6, "0057375673204660416": 6, "though": 6, "need": [6, 7, 10], "converg": [6, 7], "reach": 6, "might": [6, 7, 10], "tweak": 6, "nits_uot": [6, 15, 16], "get": 6, "fastest": 6, "rate": 6, "suptitl": [6, 7], "comparison": 6, "nsinkhorn": 6, "121": 6, "122": 6, "tight_layout": 6, "fontsiz": [6, 7], "20": [6, 7], "interpret": [6, 7], "describ": [6, 7, 19], "probability_map": [6, 7], "being": [6, 7], "ani": [6, 7], "anatomi": [6, 7, 19], "onto": [6, 7], "convers": [6, 7], "inverse_transform": [6, 7, 11, 12, 13, 17], "take": [6, 7], "predicted_target_featur": [6, 7], "predict": [6, 7], "part": [6, 7, 19], "trane": [6, 7], "realli": [6, 7], "help": [6, 7], "evalu": [6, 7, 15, 16], "captur": [6, 7], "test": [6, 7, 10], "173": [6, 8], "241": [6, 8], "plot_1_aligning_brain_dens": [6, 8], "sinc": 7, "stick": 7, "rather": 7, "around": 7, "easili": [7, 11, 17], "abov": 7, "150k": 7, "appropri": 7, "v100": 7, "nvidia": 7, "tutori": 7, "go": 7, "through": [7, 19], "explain": 7, "rope": 7, "detail": [7, 19], "current": 7, "focu": 7, "real": 7, "life": 7, "10242": 7, "fsaverage5": 7, "els": 7, "assum": 7, "m": [7, 12, 13, 15, 16, 19], "power": 7, "rule": 7, "thumb": 7, "greater": 7, "50k": 7, "therefor": 7, "cannot": 7, "explicit": 7, "under": [7, 10, 12, 13, 14, 15, 16, 17], "hood": [7, 14, 17], "randomli": 7, "higher": 7, "although": 7, "speed": 7, "rest": 7, "procedur": [7, 15, 16, 19], "so": 7, "invest": 7, "fs5_pial_left_geometry_embed": 7, "tricki": 7, "provid": [7, 19], "empir": 7, "method": [7, 11, 12, 13, 15, 16, 17, 19], "perform": 7, "oper": 7, "vector": 7, "l2": 7, "eventu": 7, "maximum": 7, "found": [7, 19], "process": [7, 12, 13, 15, 16, 19], "source_distance_max": 7, "target_distance_max": 7, "one": [7, 19], "leverag": 7, "gather": 7, "differ": [7, 12, 13, 15, 16], "Their": 7, "wa": [7, 10, 15, 16], "allow": [7, 12, 13, 19], "awai": 7, "radiu": 7, "1000": [7, 15, 16], "t0": 7, "source_sampl": 7, "target_sampl": 7, "t1": 7, "43": 7, "05": 7, "02835116721689701": 7, "02840116247534752": 7, "014070531353354454": 7, "014342601411044598": 7, "005127766635268927": 7, "0056398711167275906": 7, "002963768783956766": 7, "0036015475634485483": 7, "44": 7, "0023684604093432426": 7, "0030750904697924852": 7, "2000": 7, "1636103391647339": 7, "16446241736412048": 7, "14844009280204773": 7, "14943459630012512": 7, "14840030670166016": 7, "1494002640247345": 7, "while": [7, 19], "As": 7, "see": [7, 12, 13], "stop": [7, 15, 16], "earli": [7, 15, 16], "yet": 7, "few": 7, "good": [7, 19], "strategi": 7, "165": 7, "One": 7, "inspect": 7, "poor": 7, "some": [7, 12, 13], "area": [7, 19], "bad": 7, "consequ": [7, 19], "increas": 7, "oppos": 7, "source_sampled_surfac": 7, "target_sampled_surfac": 7, "blue": 7, "avg_method": 7, "alreadi": 7, "big": 7, "still": 7, "top": 7, "corder": 7, "coarse_pi": 7, "corner": 7, "howev": 7, "exhibit": 7, "correspond": 7, "yield": 7, "reason": 7, "diagon": 7, "quit": 7, "646302460346359": 7, "ith": 7, "row": 7, "alwai": 7, "fetch": 7, "hot": 7, "vertor": 7, "whose": [7, 13], "posit": 7, "one_hot": 7, "070": [7, 8], "728": [7, 8], "plot_2_aligning_brain_spars": [7, 8], "35": 8, "243": 8, "auto_examples_01_brain_align": 8, "brain": [8, 10, 19], "03": 8, "multipl": [10, 19], "compat": 10, "activ": [10, 19], "develop": 10, "There": 10, "guarante": 10, "chang": 10, "futur": 10, "up": 10, "date": 10, "version": 10, "pip": [10, 18], "dedic": 10, "env": 10, "git": 10, "clone": 10, "alexisthu": 10, "cd": [10, 18], "e": [10, 18], "contributor": 10, "automat": 10, "format": 10, "contribut": 10, "dev": [10, 18], "commit": 10, "configur": 10, "your": [10, 11, 17], "machin": 10, "pytest": 10, "articl": 10, "thual": [10, 19], "2022": [10, 19], "author": [10, 19], "alexi": [10, 19], "tran": [10, 19], "hui": [10, 19], "zemskova": [10, 19], "tatiana": [10, 19], "courti": [10, 19], "nicola": [10, 19], "flamari": [10, 19], "r\u00e9mi": [10, 19], "dehaen": [10, 19], "stanisla": [10, 19], "thirion": [10, 19], "bertrand": [10, 19], "publish": 10, "arxiv": [10, 19], "doi": [10, 19], "48550": [10, 19], "2206": [10, 19], "09398": [10, 19], "url": 10, "org": [10, 19], "ab": [10, 19], "year": 10, "copyright": 10, "creativ": 10, "common": 10, "attribut": 10, "modul": [11, 17], "compris": [11, 17], "main": [11, 17], "class": [11, 12, 13, 14, 15, 16, 17, 19], "appli": [11, 17, 19], "new": [11, 17], "__init__": [11, 12, 13, 14, 15, 16], "reg_mod": [12, 13, 15, 16], "joint": [12, 13, 15, 16], "init": [12, 13, 15, 16], "float": [12, 13, 15, 16], "option": [12, 13, 15, 16], "interpol": [12, 13], "rel": [12, 13], "wasserstein": [12, 13, 15, 16, 19], "gromov": [12, 13, 19], "equat": [12, 13], "tupl": [12, 13, 15, 16], "inf": [12, 13], "control": [12, 13], "margin": [12, 13, 15, 16], "constraint": [12, 13, 15, 16, 19], "forc": [12, 13], "entropi": [12, 13, 15, 16], "independ": [12, 13], "unbalanc": [12, 13, 19], "gw": [12, 13, 19], "regularis": [12, 13, 15, 16], "w": [12, 13, 15, 16, 19], "none": [12, 13, 15, 16], "source_weight": [12, 13], "target_weight": [12, 13], "init_plan": [12, 13, 15, 16], "init_du": [12, 13, 15, 16], "auto": [12, 13], "studi": [12, 13], "ndarrai": [12, 13, 15, 16], "n_featur": [12, 13], "node": [12, 13], "graph": [12, 13], "arrai": [12, 13, 15, 16], "otherwis": [12, 13], "error": [12, 13], "weight": [12, 13], "eahc": 12, "initialis": [12, 15, 16], "dual": [12, 15, 16], "potenti": 12, "basesolv": [12, 13], "param": [12, 13], "avail": [12, 13, 19], "bool": [12, 13, 15, 16], "log": [12, 13, 15, 16], "self": [12, 13], "object": [12, 13], "ot": [12, 13, 15, 16], "n_sampl": [12, 13], "transported_data": [12, 13], "x_i": 13, "x_j": 13, "coo": 13, "fugwsolv": 14, "fugw_loss": [14, 15, 16], "local_biconvex_cost": [14, 15, 16], "fugwsparsesolv": 14, "early_stopping_threshold": [15, 16], "eval_bcd": [15, 16], "eval_uot": [15, 16], "ibpp_eps_bas": [15, 16], "ibpp_nits_sinkhorn": [15, 16], "int": [15, 16], "absolut": [15, 16], "two": [15, 16, 19], "consecut": [15, 16], "threshold": [15, 16], "fall": [15, 16], "everi": [15, 16], "consid": [15, 16], "regular": [15, 16], "specif": [15, 16], "within": [15, 16], "uot": [15, 16], "gamma": [15, 16], "data_const": [15, 16], "tuple_weight": [15, 16], "hyperparam": [15, 16], "scalar": [15, 16], "combin": [15, 16], "gromow": [15, 16], "transpos": [15, 16], "cost": [15, 16], "updat": [15, 16], "Then": [15, 16], "make": [15, 16, 18], "rho_": [15, 16], "rho_t": [15, 16], "d": [15, 16, 19], "dt": [15, 16], "wt": [15, 16], "str": [15, 16], "measur": [15, 16], "assign": [15, 16], "coupl": [15, 16], "re": [15, 16], "dict": [15, 16], "dictionari": [15, 16], "kei": [15, 16], "d1": [15, 16], "d2": [15, 16], "duals_pi": [15, 16], "duals_gamma": [15, 16], "list": [15, 16], "end": [15, 16], "lower": [16, 19], "bound": [16, 19], "instal": 18, "clean": 18, "html": [18, 19], "p": 19, "denot": 19, "voxel": 19, "f_i": 19, "f_j": 19, "fuse": 19, "tri": 19, "underli": 19, "below": 19, "essenc": 19, "distant": 19, "motiv": 19, "origin": 19, "neurip": 19, "paper": 19, "work": 19, "et": 19, "al": 19, "human": 19, "cortex": 19, "behav": 19, "throughout": 19, "seri": 19, "experi": 19, "convex": 19, "understood": 19, "multitud": 19, "exist": 19, "instanc": 19, "cuturi": 19, "2013": 19, "prove": 19, "effici": 19, "unfortun": 19, "To": 19, "circumv": 19, "s\u00e9journ\u00e9": 19, "2021": 19, "reformul": 19, "bi": 19, "sum_": 19, "l": 19, "s_": 19, "t_": 19, "p_": 19, "q": 19, "q_": 19, "impos": 19, "relax": 19, "algorithm": 19, "altern": 19, "freez": 19, "resp": 19, "adapt": 19, "insid": 19, "name": 19, "classic": 19, "chizat": 19, "2017": 19, "major": 19, "chapel": 19, "inexact": 19, "bregman": 19, "proxim": 19, "xie": 19, "2020": 19, "19": 19, "june": 19, "marco": 19, "lightspe": 19, "advanc": 19, "neural": 19, "system": 19, "1306": 19, "0895": 19, "sejourn": 19, "thibault": 19, "francoi": 19, "xavier": 19, "vialard": 19, "gabriel": 19, "peyr\u00e9": 19, "conic": 19, "formul": 19, "34": 19, "8766": 19, "79": 19, "curran": 19, "associ": 19, "inc": 19, "proceed": 19, "cc": 19, "hash": 19, "4990974d150d0de5e6e15a1454fe6b0f": 19, "abstract": 19, "laetitia": 19, "haoran": 19, "wu": 19, "c\u00e9dric": 19, "f\u00e9vott": 19, "gill": 19, "gasso": 19, "neg": 19, "penal": 19, "linear": 19, "regress": 19, "23270": 19, "82": 19, "c3c617a9b80b3ae1ebd868b0017cc349": 19, "yujia": 19, "xiangfeng": 19, "wang": 19, "ruijia": 19, "hongyuan": 19, "zha": 19, "A": 19, "fast": 19, "35th": 19, "uncertainti": 19, "artifici": 19, "intellig": 19, "confer": 19, "433": 19, "53": 19, "pmlr": 19, "mlr": 19, "v115": 19, "xie20b": 19, "platt": 19, "john": 19, "fastmap": 19, "metricmap": 19, "md": 19, "nystrom": 19, "januari": 19, "2005": 19, "www": 19, "microsoft": 19, "en": 19, "research": 19, "public": 19, "lenaic": 19, "bernhard": 19, "schmitzer": 19, "fran\u00e7oi": 19, "mai": 19, "1607": 19, "05816": 19}, "objects": {"fugw.mappings": [[12, 0, 1, "", "FUGW"], [13, 0, 1, "", "FUGWSparse"]], "fugw.mappings.FUGW": [[12, 1, 1, "", "__init__"], [12, 1, 1, "", "fit"], [12, 1, 1, "", "inverse_transform"], [12, 1, 1, "", "transform"]], "fugw.mappings.FUGWSparse": [[13, 1, 1, "", "__init__"], [13, 1, 1, "", "fit"], [13, 1, 1, "", "inverse_transform"], [13, 1, 1, "", "transform"]], "fugw.solvers": [[15, 0, 1, "", "FUGWSolver"], [16, 0, 1, "", "FUGWSparseSolver"]], "fugw.solvers.FUGWSolver": [[15, 1, 1, "", "__init__"], [15, 1, 1, "", "fugw_loss"], [15, 1, 1, "", "local_biconvex_cost"], [15, 1, 1, "", "solve"]], "fugw.solvers.FUGWSparseSolver": [[16, 1, 1, "", "__init__"], [16, 1, 1, "", "fugw_loss"], [16, 1, 1, "", "local_biconvex_cost"], [16, 1, 1, "", "solve"]]}, "objtypes": {"0": "py:class", "1": "py:method"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"]}, "titleterms": {"basic": [0, 9], "transport": [1, 3], "distribut": [1, 3], "us": [1, 3, 6, 7], "dens": 1, "solver": [1, 3, 14, 17, 19], "comput": [1, 4, 6, 7, 8], "map": [1, 6, 7, 11, 17], "gener": 2, "embed": 2, "from": [2, 10], "mesh": 2, "spars": 3, "time": [4, 8], "brain": [5, 6, 7, 9], "align": [5, 6, 7, 9], "surfac": [6, 7], "2": [6, 7], "individu": [6, 7], "fmri": [6, 7], "data": [6, 7], "featur": [6, 7], "arrai": [6, 7], "geometri": [6, 7], "normal": [6, 7], "train": [6, 7], "high": 7, "resolut": 7, "estim": 7, "kernel": 7, "matric": 7, "exampl": 9, "fuse": 10, "unbalanc": 10, "gromov": 10, "wasserstein": 10, "python": 10, "instal": 10, "pypi": 10, "sourc": 10, "cite": 10, "thi": 10, "work": 10, "fugw": [11, 12, 14, 17], "fugwspars": 13, "fugwsolv": 15, "fugwsparsesolv": 16, "api": 17, "refer": [17, 19], "contribut": 18, "build": 18, "doc": 18, "introduct": 19, "optim": 19, "problem": 19}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 57}, "alltitles": {"Basics": [[0, "basics"], [9, "basics"]], "Transport distributions using dense solvers": [[1, "transport-distributions-using-dense-solvers"]], "Using the computed mapping": [[1, "using-the-computed-mapping"], [6, "using-the-computed-mapping"]], "Generate embeddings from mesh": [[2, "generate-embeddings-from-mesh"]], "Transport distributions using sparse solvers": [[3, "transport-distributions-using-sparse-solvers"]], "Computation times": [[4, "computation-times"], [8, "computation-times"]], "Brain alignment": [[5, "brain-alignment"], [9, "brain-alignment"]], "Align brain surfaces of 2 individuals with fMRI data": [[6, "align-brain-surfaces-of-2-individuals-with-fmri-data"]], "Computing feature arrays": [[6, "computing-feature-arrays"], [7, "computing-feature-arrays"]], "Computing geometry arrays": [[6, "computing-geometry-arrays"]], "Normalizing features and geometries": [[6, "normalizing-features-and-geometries"], [7, "normalizing-features-and-geometries"]], "Training the mapping": [[6, "training-the-mapping"], [7, "training-the-mapping"]], "Align high-resolution brain surfaces of 2 individuals with fMRI data": [[7, "align-high-resolution-brain-surfaces-of-2-individuals-with-fmri-data"]], "Estimating geometry kernel matrices": [[7, "estimating-geometry-kernel-matrices"]], "Using the computed mappings": [[7, "using-the-computed-mappings"]], "Examples": [[9, "examples"]], "Fused Unbalanced Gromov-Wasserstein for Python": [[10, "fused-unbalanced-gromov-wasserstein-for-python"]], "Installation": [[10, "installation"]], "From PyPI": [[10, "from-pypi"]], "From source": [[10, "from-source"]], "Citing this work": [[10, "citing-this-work"]], "fugw.mappings": [[11, "fugw-mappings"], [17, "fugw-mappings"]], "FUGW": [[12, "fugw"]], "FUGWSparse": [[13, "fugwsparse"]], "fugw.solvers": [[14, "fugw-solvers"], [17, "fugw-solvers"]], "FUGWSolver": [[15, "fugwsolver"]], "FUGWSparseSolver": [[16, "fugwsparsesolver"]], "API references": [[17, "api-references"]], "Contributing": [[18, "contributing"]], "Building the docs": [[18, "building-the-docs"]], "Introduction": [[19, "introduction"]], "Optimization problem": [[19, "optimization-problem"]], "Solvers": [[19, "solvers"]], "References": [[19, "references"]]}, "indexentries": {"fugw (class in fugw.mappings)": [[12, "fugw.mappings.FUGW"]], "__init__() (fugw.mappings.fugw method)": [[12, "fugw.mappings.FUGW.__init__"]], "fit() (fugw.mappings.fugw method)": [[12, "fugw.mappings.FUGW.fit"]], "inverse_transform() (fugw.mappings.fugw method)": [[12, "fugw.mappings.FUGW.inverse_transform"]], "transform() (fugw.mappings.fugw method)": [[12, "fugw.mappings.FUGW.transform"]], "fugwsparse (class in fugw.mappings)": [[13, "fugw.mappings.FUGWSparse"]], "__init__() (fugw.mappings.fugwsparse method)": [[13, "fugw.mappings.FUGWSparse.__init__"]], "fit() (fugw.mappings.fugwsparse method)": [[13, "fugw.mappings.FUGWSparse.fit"]], "inverse_transform() (fugw.mappings.fugwsparse method)": [[13, "fugw.mappings.FUGWSparse.inverse_transform"]], "transform() (fugw.mappings.fugwsparse method)": [[13, "fugw.mappings.FUGWSparse.transform"]], "fugwsolver (class in fugw.solvers)": [[15, "fugw.solvers.FUGWSolver"]], "__init__() (fugw.solvers.fugwsolver method)": [[15, "fugw.solvers.FUGWSolver.__init__"]], "fugw_loss() (fugw.solvers.fugwsolver method)": [[15, "fugw.solvers.FUGWSolver.fugw_loss"]], "local_biconvex_cost() (fugw.solvers.fugwsolver method)": [[15, "fugw.solvers.FUGWSolver.local_biconvex_cost"]], "solve() (fugw.solvers.fugwsolver method)": [[15, "fugw.solvers.FUGWSolver.solve"]], "fugwsparsesolver (class in fugw.solvers)": [[16, "fugw.solvers.FUGWSparseSolver"]], "__init__() (fugw.solvers.fugwsparsesolver method)": [[16, "fugw.solvers.FUGWSparseSolver.__init__"]], "fugw_loss() (fugw.solvers.fugwsparsesolver method)": [[16, "fugw.solvers.FUGWSparseSolver.fugw_loss"]], "local_biconvex_cost() (fugw.solvers.fugwsparsesolver method)": [[16, "fugw.solvers.FUGWSparseSolver.local_biconvex_cost"]], "solve() (fugw.solvers.fugwsparsesolver method)": [[16, "fugw.solvers.FUGWSparseSolver.solve"]]}})