from thumt.utils.hparams import HParams
from thumt.utils.inference import beam_search, argmax_decoding
from thumt.utils.evaluation import evaluate
from thumt.utils.checkpoint import save, latest_checkpoint
from thumt.utils.scope import scope, get_scope, unique_name
from thumt.utils.misc import get_global_step, set_global_step
from thumt.utils.misc import get_global_time, set_global_time
from thumt.utils.utils import param_in, get_reg_loss, dim_dropout, compute_common_score
from thumt.utils.convert_params import params_to_vec, vec_to_params
from thumt.utils.head_utils import visualize_head_selection, head_importance_score
from thumt.utils.head_utils import prune_linear_layer, prune_head_vector, prune_vector
from thumt.utils.head_utils import reinit_linear_layer, reinit_vector_
from thumt.utils.head_utils import reverse_select
from thumt.utils.head_utils import find_pruneable_heads_and_indices, find_pruneable_heads_indices
from thumt.utils.head_utils import selected_linear
from thumt.utils.head_utils import headwise_mask
