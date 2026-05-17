from features import get_train_test_data
from itertools import product
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import regularizers
import ctypes  # preloaded by .pth, but good practice
import numpy as np
import pandas as pd
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


LOSS_FUNCTION = "binary_crossentropy"
TARGET_AUC = 0.85

# ── Grid ──────────────────────────────────────────────────────────────────────
PARAM_GRID = {
    "learning_rate": [0.001, 0.005],
    "dropout_rate": [0.1, 0.3],
    "l2_reg": [1e-9, 1e-4],
    "batch_size": [128, 256],
    "epochs": [2],
}

# ── Scoreboard ────────────────────────────────────────────────────────────────
COLS = {
    "#": 4,
    "lr": 7,
    "dropout": 9,
    "l2": 8,
    "batch": 7,
    "epochs": 7,
    "val_auc": 9,
    "status": 16,
}

BOARD_WIDTH = sum(COLS.values()) + len(COLS) + 1  # separators
# Total printed lines = top border + header + divider + n rows + bottom border + status line
BOARD_LINES = 5 + 0  # will add n_combos dynamically after combos are built

SCOREBOARD_FILE = "scoreboard.txt"


# ── Model factory ─────────────────────────────────────────────────────────────
def create_model(input_dim, dropout_rate=0.1, l2_reg=1e-9):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for _ in range(4):
        x = tf.keras.layers.Dense(
            512, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _row(vals: dict, widths: dict) -> str:
    cells = [f"{str(v):<{w}}" for v, w in zip(vals.values(), widths.values())]
    return "│" + "│".join(cells) + "│"


def _divider(char="─") -> str:
    return "├" + "┼".join(char * w for w in COLS.values()) + "┤"


def _border(top=True) -> str:
    inner = "┬".join("─" * w for w in COLS.values())
    return (
        ("╔" + inner.replace("─", "═").replace("┬", "╦") + "╗")
        if top
        else ("╚" + inner.replace("─", "═").replace("┬", "╩") + "╝")
    )


def render_scoreboard(
    rows: list[dict], epoch: int, cur_auc: float, run_idx: int, total: int
):
    """Print the full scoreboard. Call once to init, then overwrite in-place."""
    lines = []
    lines.append(_border(top=True))
    lines.append(
        _row(
            {
                "#": "#",
                "lr": "lr",
                "dropout": "dropout",
                "l2": "l2",
                "batch": "batch",
                "epochs": "epochs",
                "val_auc": "val_auc",
                "status": "status",
            },
            COLS,
        )
    )
    lines.append(_divider())
    for r in rows:
        lines.append(_row(r, COLS))
    lines.append(_border(top=False))
    lines.append(
        f"  run [{run_idx}/{total}] | epoch {epoch:>4} | "
        f"cur val_auc: {cur_auc:.4f} | target: {TARGET_AUC}"
    )
    print("\n".join(lines))


class ScoreboardCallback(tf.keras.callbacks.Callback):
    def __init__(self, target_auc, run_idx, total, params, scoreboard_rows):
        super().__init__()
        self.target_auc = target_auc
        self.run_idx = run_idx
        self.total = total
        self.params = params
        self.scoreboard_rows = scoreboard_rows
        self.best_so_far = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_auc = float(logs.get("val_auc", 0.0))
        if val_auc > self.best_so_far:
            self.best_so_far = val_auc

        row = self.scoreboard_rows[self.run_idx - 1]
        row["val_auc"] = f"{self.best_so_far:.4f}"
        row["status"] = f"▶ ep {epoch + 1}"

        display = sorted(
            self.scoreboard_rows,
            key=lambda r: float(r["val_auc"]) if r["val_auc"] != "—" else -1,
            reverse=True,
        )

        self._write(display, epoch + 1, val_auc)

        if val_auc >= self.target_auc:
            print(f"\n  🎯 Target AUC {self.target_auc} reached!")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        row = self.scoreboard_rows[self.run_idx - 1]
        row["status"] = "✔ done" if self.best_so_far < self.target_auc else "🎯 hit"

    def _write(self, rows, epoch, cur_auc):
        lines = []
        lines.append(_border(top=True))
        lines.append(
            _row(
                {
                    "#": "#",
                    "lr": "lr",
                    "dropout": "dropout",
                    "l2": "l2",
                    "batch": "batch",
                    "epochs": "epochs",
                    "val_auc": "val_auc",
                    "status": "status",
                },
                COLS,
            )
        )
        lines.append(_divider())
        for r in rows:
            lines.append(_row(r, COLS))
        lines.append(_border(top=False))
        lines.append(
            f"  run [{self.run_idx}/{self.total}] | epoch {epoch:>4} | "
            f"cur val_auc: {cur_auc:.4f} | target: {TARGET_AUC}"
        )
        with open(SCOREBOARD_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

# ── Notebook callback ─────────────────────────────────────────────────────────
class NotebookCallback(tf.keras.callbacks.Callback):
    def __init__(self, target_auc, global_best, best_model_path, params=None):
        super().__init__()
        self.target_auc      = target_auc
        self.global_best     = global_best
        self.best_model_path = best_model_path
        self.params          = params or {}
        self.best_so_far     = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_auc  = float(logs.get("val_auc",  0.0))
        val_loss = float(logs.get("val_loss", 0.0))
        auc      = float(logs.get("auc",      0.0))
        loss     = float(logs.get("loss",     0.0))

        if val_auc > self.best_so_far:
            self.best_so_far = val_auc
            if val_auc > self.global_best["auc"]:
                self.global_best["auc"] = val_auc
                self.model.save(self.best_model_path)

        print(
            f"  ep {epoch+1:>4} | "
            f"loss: {loss:.4f}  auc: {auc:.4f} | "
            f"val_loss: {val_loss:.4f}  val_auc: {val_auc:.4f}  best: {self.best_so_far:.4f}",
            end="\r",
        )

        if val_auc >= self.target_auc:
            print(f"\n  🎯 Target AUC {self.target_auc} reached at epoch {epoch + 1}!")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        print()   # newline after the last \r
if __name__ == "__main__":
    # get data
    data = get_train_test_data(
        "data/bank_data_train.csv",
        sampler="none",
        num_impute="mean",
        cat_handle="none",
        cat_k=1,
        threshold_missing_data=0.8,
    )

    print(f"x_train shape: {data.x_train.shape}")
    print(f"x_test shape: {data.x_test.shape}")
    print(f"y_train shape: {data.y_train.shape}")
    print(f"y_test shape: {data.y_test.shape}")

    # Display first 5 rows of x_train
    data.x_train.head()

    # ── Class weights ─────────────────────────────────────────────────────────────
    weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(data.y_train), y=data.y_train
    )
    class_weights = dict(enumerate(weights))
    print(f"Class weights: {class_weights}\n")

    # ── Grid search ───────────────────────────────────────────────────────────────
    keys = list(PARAM_GRID.keys())
    combos = list(product(*PARAM_GRID.values()))
    total = len(combos)

    BOARD_LINES = 5 + total  # top + header + divider + n rows + bottom + status

    # Pre-build all scoreboard rows as "pending"
    scoreboard_rows = [
        {
            "#": str(i + 1),
            "lr": str(vals[keys.index("learning_rate")]),
            "dropout": str(vals[keys.index("dropout_rate")]),
            "l2": str(vals[keys.index("l2_reg")]),
            "batch": str(vals[keys.index("batch_size")]),
            "epochs": str(vals[keys.index("epochs")]),
            "val_auc": "—",
            "status": "pending",
        }
        for i, vals in enumerate(combos)
    ]

    # Initial render
    render_scoreboard(scoreboard_rows, epoch=0, cur_auc=0.0, run_idx=0, total=total)
    target_hit = False

    for run_idx, values in enumerate(combos, 1):
        params = dict(zip(keys, values))

        # Mark current row as running
        scoreboard_rows[run_idx - 1]["status"] = "▶ running"

        tf.keras.backend.clear_session()

        model = create_model(
            input_dim=data.x_train.shape[1],
            dropout_rate=params["dropout_rate"],
            l2_reg=params["l2_reg"],
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
            loss=LOSS_FUNCTION,
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )

        callback = ScoreboardCallback(
            target_auc=TARGET_AUC,
            run_idx=run_idx,
            total=total,
            params=params,
            scoreboard_rows=scoreboard_rows,
            # board_lines     = BOARD_LINES,
        )

        model.fit(
            data.x_train,
            data.y_train,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=[callback],
            validation_data=(data.x_test, data.y_test),
            class_weight=class_weights,
            verbose=0,
        )

        scoreboard_rows[run_idx - 1]["val_auc"] = f"{callback.best_so_far:.4f}"

        if callback.best_so_far >= TARGET_AUC:
            target_hit = True
            break
