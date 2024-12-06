""" Enums. """
from enum import Enum


class OrderedEnum(Enum):
    # pylint: disable=comparison-with-callable
    """ OrderedEnum backport.

    :meta private:
    """
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Direction(OrderedEnum):
    """ Connection directions. """
    AFFERENT = "afferent"
    EFFERENT = "efferent"


class Cell:
    """ Cell property names. """
    ID = "gid"

    MORPHOLOGY = "morphology"
    ME_COMBO = "me_combo"
    MTYPE = "mtype"
    ETYPE = "etype"
    LAYER = "layer"
    REGION = "region"
    HYPERCOLUMN = "hypercolumn"  #: Legacy property
    MINICOLUMN = "minicolumn"  #: Legacy property
    MORPH_CLASS = "morph_class"
    SYNAPSE_CLASS = "synapse_class"
    X = "x"
    Y = "y"
    Z = "z"
    ORIENTATION = "orientation"


class Synapse(OrderedEnum):
    """ Synapse property names. """
    PRE_GID = "pre_gid"
    POST_GID = "post_gid"

    #: axonal delay (milliseconds)
    AXONAL_DELAY = "axonal_delay"
    #: The depression time constant of the synapse (milliseconds)
    D_SYN = "d_syn"
    #: The decay time of the synapse (milliseconds).
    DTC = "DTC"
    #: The facilitation time constant (milliseconds) of the synapse.
    F_SYN = "f_syn"
    #: The conductance of the synapse (nanosiemens).
    G_SYNX = "g_synx"
    #: Number of readily releasable pool of vesicles.
    NRRP = "NRRP"
    #: synapse type: Inhibitory < 100 or Excitatory >= 100 (see syn_type_id in SONATA spec)
    TYPE = "type"
    #: The u parameter in the Tsodyks Markram Model.
    U_SYN = "u_syn"

    PRE_BRANCH_ORDER = "pre_branch_order"
    PRE_NEURITE_DISTANCE = "pre_neurite_distance"  #: Path length to soma from synapse
    PRE_SECTION_DISTANCE = "pre_section_distance"  #: Path length within the section
    #: Segment considered by NEURON for the synapse
    #: Note: this differs from MorphIO's ID scheme, since the soma isn't counted by MorphIO
    PRE_SECTION_ID = "pre_section_id"
    PRE_SEGMENT_ID = "pre_segment_id"  #: Segment ID within the PRE_SECTION_ID
    PRE_SEGMENT_OFFSET = "pre_segment_offset"  #: Offset within the PRE_SEGMENT_ID, in um

    POST_BRANCH_ORDER = "post_branch_order"
    POST_BRANCH_TYPE = "post_branch_type"

    POST_NEURITE_DISTANCE = "post_neurite_distance"  #: Path length to soma from synapse
    POST_SECTION_DISTANCE = "post_section_distance"  #: Path length within the section
    #: Segment considered by NEURON for the synapse
    #: Note: this differs from MorphIO's ID scheme, since the soma isn't counted by MorphIO
    POST_SECTION_ID = "post_section_id"
    POST_SEGMENT_ID = "post_segment_id"  #: Segment ID within the POST_SECTION_ID
    POST_SEGMENT_OFFSET = "post_segment_offset"  #: Offset within the POST_SEGMENT_ID, in um

    PRE_X_CENTER = "pre_x_center"  #: presynaptic touch position (in the center axis of segment)
    PRE_Y_CENTER = "pre_y_center"  #: presynaptic touch position (in the center axis of segment)
    PRE_Z_CENTER = "pre_z_center"  #: presynaptic touch position (in the center axis of segment)

    PRE_X_CONTOUR = "pre_x_contour"  #: presynaptic touch position (on the segment surface)
    PRE_Y_CONTOUR = "pre_y_contour"  #: presynaptic touch position (on the segment surface)
    PRE_Z_CONTOUR = "pre_z_contour"  #: presynaptic touch position (on the segment surface)

    POST_X_CENTER = "post_x_center"  #: postsynaptic touch position (in the center axis of segment)
    POST_Y_CENTER = "post_y_center"  #: postsynaptic touch position (in the center axis of segment)
    POST_Z_CENTER = "post_z_center"  #: postsynaptic touch position (in the center axis of segment)

    POST_X_CONTOUR = "post_x_contour"  #: postsynaptic touch position (on the segment surface)
    POST_Y_CONTOUR = "post_y_contour"  #: postsynaptic touch position (on the segment surface)
    POST_Z_CONTOUR = "post_z_contour"  #: postsynaptic touch position (on the segment surface)

    TOUCH_DISTANCE = "touch_distance"  #: distance between the two PRE and POST CONTOUR positions

    # sonata only
    #: The scale factor for the conductance (no unit).If no value or negative, no change is applied.
    CONDUCTANCE_RATIO = "conductance_scale_factor"
    #: A coefficient describing the scaling of u to be done by the simulator.
    #: See sonata documentation
    U_HILL_COEFFICIENT = "u_hill_coefficient"


class Section(OrderedEnum):
    """ Section property/feature names. """
    ID = "section_id"

    NEURITE_TYPE = "neurite_type"
    BRANCH_ORDER = "branch_order"
    LENGTH = "length"
    NEURITE_START_DISTANCE = "neurite_start_distance"


class Segment(OrderedEnum):
    """ Segment property/feature names. """
    ID = "segment_id"

    X1 = "x1"
    Y1 = "y1"
    Z1 = "z1"
    R1 = "r1"
    X2 = "x2"
    Y2 = "y2"
    Z2 = "z2"
    R2 = "r2"

    LENGTH = 'length'
    SECTION_START_DISTANCE = 'section_start_distance'

    VOLUME = 'volume'
    REGION = 'region'
