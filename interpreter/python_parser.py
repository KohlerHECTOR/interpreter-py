from sklearn import tree
import numpy as np
import ast


def parse_to_python(
    clf,
    env,
    file_name="play.py",
    feature_names=None,
    action_names=None,
    is_oblique=True,
):
    if isinstance(clf, tree.DecisionTreeRegressor):
        kw_ = "value"
    elif isinstance(clf, tree.DecisionTreeClassifier):
        kw_ = "class"

    if feature_names == None:
        feature_names = [
            "feat_" + str(i) for i in range(env.observation_space.shape[0])
        ]

    s = ["{}".format(i) for i in np.array(feature_names)]
    s = np.array(s)

    if is_oblique:
        s_ = [" - {}".format(i) for i in np.array(feature_names)]
        s_ = np.array(s_)
        s_mat = np.tile(s, (s.shape[0], 1))
        s_mat_ = np.tile(s_, (s_.shape[0], 1))
        # pint(s_mat_)

        diff_s = []
        for m in range(s_mat.shape[0]):
            level = []
            for j in range(s_mat.shape[1]):
                level.append(s_mat[m, j] + s_mat_[j, m])
            diff_s.append(level)

        diff_s = np.array(diff_s, dtype=np.str_)

        s_comb = np.append(s, diff_s[np.tril_indices(s.shape[0], k=-1)])
    else:
        s_comb = s

    if action_names == None:
        class_names = [str(i) for i in range(env.action_space.n)]
    else:
        class_names = action_names

    r = tree.export_text(clf, feature_names=s_comb, class_names=class_names)
    print(r)
    with open(file_name, "a") as the_file:
        the_file.write("def play(state):\n")
        for line in r.split("\n")[:-1]:
            split_indent = line.split("|")
            nb_indent = 2 * (
                len(split_indent) - 2 + 1
            )  # first empty last is if else + 1for def
            features_sign_val = split_indent[-1].split("--- ")[1]
            if "<=" in features_sign_val:
                each_feat_val = features_sign_val.split("<=")

                featss = each_feat_val[0]
                if "-" in featss:
                    each_feat = each_feat_val[0].split(" - ")
                    val = each_feat_val[1]

                    python_line = (
                        nb_indent * "  "
                        + "if state."
                        + each_feat[0]
                        + " - "
                        + "state."
                        + each_feat[1]
                        + " <="
                        + val
                        + ":\n"
                    )
                else:
                    python_line = (
                        nb_indent * "  " + "if state." + features_sign_val + ":\n"
                    )
            elif ">" in features_sign_val:
                python_line = nb_indent * "  " + "else:\n"
            else:
                action_vec_str = features_sign_val.split("{}: ".format(kw_))[1]
                x = ast.literal_eval(action_vec_str)
                if isinstance(x, int):
                    python_line = nb_indent * "  " + "return " + class_names[x] + "\n"
                else:
                    returned_action = "{"
                    for k, val in enumerate(x):
                        returned_action += (
                            '"' + class_names[k] + '": ' + str(val) + ", "
                        )
                    python_line = nb_indent * "  " + "return " + returned_action + "}\n"
            the_file.write(python_line)
