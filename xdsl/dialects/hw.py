"""
This dialect defines the `hw` dialect, which is intended to be a generic
representation of HW outside of a particular use-case.

[1] https://circt.llvm.org/docs/Dialects/HW/
"""
from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from enum import Enum
from typing import Annotated, Generic, TypeVar

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    ArrayAttr,
    DictionaryAttr,
    FlatSymbolRefAttr,
    IntegerAttr,
    IntegerType,
    LocationAttr,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
    i64,
)
from xdsl.dialects.utils import parse_return_op_like, print_return_op_like
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    Data,
    Dialect,
    Operation,
    OpResult,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrConstraint,
    ConstraintVar,
    IRDLOperation,
    Operand,
    ParameterDef,
    VarOperand,
    VarOpResult,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import (
    ConstantLike,
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    OpTrait,
    Pure,
    SingleBlockImplicitTerminator,
    SymbolOpInterface,
    ensure_terminator,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

################################################################################
### Traits and Interfaces
################################################################################


class InnerRefNamespace(OpTrait):
    """
    Namespace to track the names used by inner symbols within an
    InnerSymbolTable.
    """


class InnerSymbolTable(OpTrait):
    """A table of inner symbols and their resolutions."""


class HWModuleLike(OpTrait):
    """Provide common module information."""

    @classmethod
    def get_hw_module_type(cls) -> ModuleType:
        raise Exception("interface method not implemented")

    @classmethod
    def set_hw_module_type(cls, type: ModuleType) -> None:
        raise Exception("interface method not implemented")

    @classmethod
    def get_all_port_attrs(cls) -> Sequence[Attribute]:
        raise Exception("interface method not implemented")

    @classmethod
    def set_all_port_attrs(cls, attrs: Sequence[Attribute]) -> None:
        raise Exception("interface method not implemented")

    @classmethod
    def remove_all_port_attrs(cls) -> None:
        raise Exception("interface method not implemented")

    @classmethod
    def get_all_port_locs(cls) -> Sequence[LocationAttr]:
        raise Exception("interface method not implemented")

    @classmethod
    def set_all_port_locs(cls, locs: Sequence[LocationAttr]) -> None:
        raise Exception("interface method not implemented")

    @classmethod
    def set_all_port_names(cls, names: Sequence[Attribute]) -> None:
        raise Exception("interface method not implemented")


class PortList(OpTrait):
    """Operations which produce a unified port list representation"""

    @classmethod
    def get_port_list(cls) -> Sequence[PortAttr]:
        raise Exception("interface method not implemented")

    @classmethod
    def get_port(cls, index: int) -> PortAttr:
        raise Exception("interface method not implemented")

    @classmethod
    def get_port_id_for_input_id(cls, index: int) -> int:
        raise Exception("interface method not implemented")

    @classmethod
    def get_port_id_for_output_id(cls, index: int) -> int:
        raise Exception("interface method not implemented")

    @classmethod
    def get_num_ports(cls) -> int:
        raise Exception("interface method not implemented")

    @classmethod
    def get_num_input_ports(cls) -> int:
        raise Exception("interface method not implemented")

    @classmethod
    def get_num_output_ports(cls) -> int:
        raise Exception("interface method not implemented")


class HWInstanceLike(OpTrait):
    """Provide common instance information."""


class InnerRefUser(OpTrait):
    """
    This interface describes an operation that may use a `InnerRef`. This
    interface allows for users of inner symbols to hook into verification and
    other inner symbol related utilities that are either costly or otherwise
    disallowed within a traditional operation.
    """

    @classmethod
    def verifyInnerRefs(cls, ns: InnerRefNamespace):
        raise Exception("interface method not implemented")


class InnerSymbol(OpTrait):
    """
    This interface describes an operation that may define an `inner_sym`.  An
    `inner_sym` operation resides in arbitrarily-nested regions of a region that
    defines a `InnerSymbolTable`.
    Inner Symbols are different from normal symbols due to MLIR symbol table
    resolution rules.  Specifically normal symbols are resolved by first going
    up to the closest parent symbol table and resolving from there (recursing
    down for complex symbol paths).  In HW and SV, modules define a symbol in a
    circuit or std.module symbol table.  For instances to be able to resolve the
    modules they instantiate, the symbol use in an instance must resolve in the
    top-level symbol table.  If a module were a symbol table, instances
    resolving a symbol would start from their own module, never seeing other
    modules (since resolution would start in the parent module of the instance
    and be unable to go to the global scope).  The second problem arises from
    nesting.  Symbols defining ops must be immediate children of a symbol table.
    HW and SV operations which define a inner_sym are grandchildren, at least,
    of a symbol table and may be much further nested.  Lastly, ports need to
    define inner_sym, something not allowed by normal symbols.

    Any operation implementing an InnerSymbol may have the inner symbol be
    optional and all methods should be robuse to the attribute not being
    defined.
    """

    @classmethod
    def getInnerNameAttr(cls) -> StringAttr:
        raise Exception("interface method not implemented")

    @classmethod
    def getInnerName(cls) -> str | None:
        raise Exception("interface method not implemented")

    @classmethod
    def setInnerSymbol(cls, name: StringAttr):
        raise Exception("interface method not implemented")

    @classmethod
    def setInnerSymbolAttr(cls, sym: InnerSymAttr):
        raise Exception("interface method not implemented")

    @classmethod
    def getInnerRef(cls) -> InnerRefAttr:
        raise Exception("interface method not implemented")

    @classmethod
    def getInnerSymAttr(cls) -> InnerSymAttr:
        raise Exception("interface method not implemented")

    @classmethod
    def supportsPerFieldSymbols(cls) -> bool:
        raise Exception("interface method not implemented")

    @classmethod
    def getTargetResultIndex(cls) -> int | None:
        raise Exception("interface method not implemented")

    @classmethod
    def getTargetResult(cls) -> OpResult:
        raise Exception("interface method not implemented")

    def verify(self, op: Operation) -> None:
        pass


################################################################################
### Attributes
################################################################################


class Direction(Enum):
    """Enum representing a module port direction"""

    INPUT = "in"
    INOUT = "inout"
    OUTPUT = "out"


@irdl_attr_definition
class DirectionAttr(Data[Direction]):
    """
    Attribute representing a module port direction.

    Note that this attribute does not exist in CIRCT, instead a native C++
    struct is used there.
    """

    name = "hw.direction"

    def __init__(self, dir: Direction):
        super().__init__(dir)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> Direction:
        for option in Direction:
            if parser.parse_optional_characters(option.value) is not None:
                return option

        raise Exception("unsupported direction string")

    def print_parameter(self, printer: Printer):
        printer.print(self.data.value)


@irdl_attr_definition
class PortAttr(ParametrizedAttribute):
    """
    Attribute representing the static information of a single module port.

    Note that this attribute does not exist in CIRCT, instead a native C++
    struct is used there.
    """

    name = "hw.port"

    port_dir: ParameterDef[DirectionAttr]
    port_name: ParameterDef[StringAttr]
    port_type: ParameterDef[Attribute]

    def __init__(self, dir: Direction, name: StringAttr, type: Attribute):
        return super().__init__([DirectionAttr(dir), name, type])

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        dir = DirectionAttr.parse_parameter(parser)
        name = parser.parse_str_literal()
        parser.parse_punctuation(":")
        type = parser.parse_type()
        return [DirectionAttr(dir), StringAttr(name), type]

    def print_parameters(self, printer: Printer) -> None:
        self.port_dir.print_parameter(printer)
        printer.print(" ")
        printer.print(self.port_name)
        printer.print(" : ")
        printer.print(self.port_type)


@irdl_attr_definition
class StructFieldAttr(ParametrizedAttribute):
    """
    Attribute representing a single field of a StructType.

    Note that this attribute does not exist in CIRCT, instead a native C++
    struct is used there.
    """

    name = "hw.struct.field"

    field_name: ParameterDef[StringAttr]
    field_type: ParameterDef[Attribute]

    def __init__(self, field_name: str, field_type: Attribute):
        super().__init__([StringAttr(field_name), field_type])

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        name = parser.parse_identifier()
        parser.parse_punctuation(":")
        type = parser.parse_type()
        return [StringAttr(name), type]

    def print_parameters(self, printer: Printer):
        printer.print(self.field_name.data)
        printer.print(": ")
        printer.print(self.field_type)


@irdl_attr_definition
class ParamDeclAttr(ParametrizedAttribute):
    """
    Module or instance parameter definition.

    An attribute describing a module parameter, or instance parameter
    specification.
    """

    name = "hw.param.decl"

    param_name: ParameterDef[StringAttr]
    param_type: ParameterDef[Attribute]
    param_value: ParameterDef[Attribute]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_punctuation("<")
        name = parser.parse_str_literal()
        parser.parse_punctuation(":")
        type = parser.parse_type()
        value = Attribute()
        if parser.parse_optional_punctuation("="):
            value = parser.parse_attribute()
        parser.parse_punctuation(">")
        return [StringAttr(name), type, value]

    def print_parameters(self, printer: Printer) -> None:
        printer.print("<")
        printer.print_attribute(self.param_name)
        printer.print(" : ")
        printer.print_attribute(self.param_type)
        if self.param_value:
            printer.print(" = ")
            printer.print_attribute(self.param_value)
        printer.print(">")


@irdl_attr_definition
class InnerRefAttr(ParametrizedAttribute):
    """
    Refer to a name inside a module.

    This works like a symbol reference, but to a name inside a module.
    """

    name = "hw.innerNameRef"

    module_ref: ParameterDef[FlatSymbolRefAttr]
    ref_name: ParameterDef[StringAttr]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_punctuation("<")
        attr = parser.parse_attribute()
        if not isinstance(attr, SymbolRefAttr):
            raise Exception(f"expected SymbolRefAttr, but got {attr}")
        parser.parse_punctuation(">")
        if len(attr.nested_references.data) != 1:
            return [Attribute(), Attribute()]
        return [attr.root_reference, attr.nested_references.data[0]]

    def print_parameters(self, printer: Printer) -> None:
        printer.print("<@")
        printer.print(self.module_ref.root_reference.data)
        printer.print("::@")
        printer.print(self.ref_name.data)
        printer.print(">")


@irdl_attr_definition
class InnerSymPropertiesAttr(ParametrizedAttribute):
    name = "hw.innerSymProps"

    sym_name: ParameterDef[StringAttr]
    field_id: ParameterDef[AnyIntegerAttr]
    sym_visibility: ParameterDef[StringAttr]

    def __init__(
        self,
        name: StringAttr,
        id: AnyIntegerAttr = IntegerAttr.from_int_and_width(0, 64),
        visibility: StringAttr = StringAttr("public"),
    ):
        super().__init__([name, id, visibility])

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_punctuation("<")
        name = parser.parse_symbol_name()
        parser.parse_punctuation(",")
        id = parser.parse_integer()
        parser.parse_punctuation(",")
        visibility = parser.parse_optional_keyword("public")
        if visibility is None:
            visibility = parser.parse_optional_keyword("private")
        if visibility is None:
            visibility = parser.parse_keyword("nested")

        return [name, IntegerAttr(id, IntegerType(64)), StringAttr(visibility)]

    def print_parameters(self, printer: Printer) -> None:
        printer.print(
            f"<@{self.sym_name.data}, {self.field_id.value.data}, {self.sym_visibility.data}>"
        )

    def verify(self):
        if len(self.sym_name.data) <= 0:
            raise VerifyException("inner symbol cannot have empty name")
        if (
            not isinstance(self.field_id.type, IntegerType)
            or self.field_id.type.width != 64
        ):
            raise VerifyException("field id must be a 64-bit integer")
        if self.sym_visibility.data not in ["public", "private", "nested"]:
            raise VerifyException(
                "visibility must be either 'public', 'private', or 'nested'"
            )


@irdl_attr_definition
class InnerSymAttr(ParametrizedAttribute):
    """
    Inner symbol definition.

    Defines the properties of an inner_sym attribute. It specifies the symbol
    name and symbol visibility for each field ID. For any ground types, there
    are no subfields and the field ID is 0. For aggregate types, a unique field
    ID is assigned to each field by visiting them in a depth-first pre-order.
    The custom assembly format ensures that for ground types, only `@<sym_name>`
    is printed.
    """

    name = "hw.innerSym"

    props: ParameterDef[ArrayAttr[InnerSymPropertiesAttr]]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_punctuation("<")
        attr = parser.parse_attribute()
        if not isinstance(attr, SymbolRefAttr):
            raise Exception(f"expected SymbolRefAttr, but got {attr}")
        parser.parse_punctuation(">")
        if len(attr.nested_references.data) != 1:
            return [Attribute(), Attribute()]
        return [attr.root_reference, attr.nested_references.data[0]]

    def print_parameters(self, printer: Printer) -> None:
        printer.print("<@")
        printer.print(self.module_ref.root_reference.data)
        printer.print("::@")
        printer.print(self.ref_name.data)
        printer.print(">")


################################################################################
### Types
################################################################################


@irdl_attr_definition
class StructType(ParametrizedAttribute, TypeAttribute):
    """Represents a structure of name, value pairs"""

    name = "hw.struct"

    fields: ParameterDef[ArrayAttr[StructFieldAttr]]

    def __init__(self, fields: Sequence[StructFieldAttr]):
        super().__init__([ArrayAttr(fields)])

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        def parse_field():
            name = parser.parse_identifier()
            parser.parse_punctuation(":")
            type = parser.parse_type()
            return StructFieldAttr(name, type)

        fields = parser.parse_comma_separated_list(Parser.Delimiter.ANGLE, parse_field)
        return [ArrayAttr(fields)]

    def print_parameters(self, printer: Printer):
        printer.print("<")
        printer.print_list(self.fields.data, lambda f: f.print_parameters(printer))
        printer.print(">")


@irdl_attr_definition
class EnumType(ParametrizedAttribute, TypeAttribute):
    """
    Represents an enumeration of values. Enums are interpreted as integers with
    a synthesis-defined encoding.
    """

    name = "hw.enum"

    fields: ParameterDef[ArrayAttr[StringAttr]]

    def __init__(self, fields: Sequence[StringAttr]):
        super().__init__([ArrayAttr(fields)])

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        def parse_field():
            name = parser.parse_identifier()
            return StringAttr(name)

        fields = parser.parse_comma_separated_list(Parser.Delimiter.ANGLE, parse_field)
        return [ArrayAttr(fields)]

    def print_parameters(self, printer: Printer):
        printer.print("<")
        printer.print_list(self.fields.data, lambda f: printer.print(f.data))
        printer.print(">")


@irdl_attr_definition
class ModuleType(ParametrizedAttribute, TypeAttribute):
    """The ModuleType contains the port information of a HWModule"""

    name = "hw.modty"

    ports: ParameterDef[ArrayAttr[PortAttr]]

    def __init__(self, ports: Sequence[PortAttr]):
        return super().__init__([ArrayAttr(ports)])

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        def parse_port() -> Attribute:
            dir = DirectionAttr.parse_parameter(parser)
            name = parser.parse_str_literal()
            parser.parse_punctuation(":")
            type = parser.parse_type()
            return PortAttr(dir, StringAttr(name), type)

        ports = parser.parse_comma_separated_list(parser.Delimiter.ANGLE, parse_port)
        return [ArrayAttr(ports)]

    def print_parameters(self, printer: Printer) -> None:
        printer.print("<")
        printer.print_list(self.ports, lambda port: port.print_parameters(printer))
        printer.print(">")


_ElementType = TypeVar("_ElementType", bound=Attribute, covariant=True)


class ArrayTypeBase(Generic[_ElementType], ParametrizedAttribute, TypeAttribute, ABC):
    size: ParameterDef[Attribute]
    element_type: ParameterDef[_ElementType]

    def __init__(self, size: int, element_type: _ElementType):
        return super().__init__([IntegerAttr(size, i64), element_type])

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_punctuation("<")
        size = parser.parse_shape_dimension()
        parser.parse_shape_delimiter()
        elem_type = parser.parse_type()
        parser.parse_punctuation(">")
        return [IntegerAttr(size, i64), elem_type]

    def print_parameters(self, printer: Printer) -> None:
        assert isinstance(self.size, IntegerAttr)
        printer.print(f"<{self.size.value.data}x{self.element_type}>")


@irdl_attr_definition
class ArrayType(Generic[_ElementType], ArrayTypeBase[_ElementType]):
    """ArrayType
    Fixed sized HW arrays are roughly similar to C arrays. On the wire (vs.
    in a memory), arrays are always packed. Memory layout is not defined as
    it does not need to be since in silicon there is not implicit memory
    sharing.
    """

    name = "hw.array"


@irdl_attr_definition
class UnpackedArrayType(Generic[_ElementType], ArrayTypeBase[_ElementType]):
    """ArrayType
    Unpacked arrays are a more flexible array representation than packed arrays,
    and are typically used to model memories. See SystemVerilog Spec 7.4.2.
    """

    name = "hw.uarray"


################################################################################
### Constraints
################################################################################


class HWAggregateType(AttrConstraint):
    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, ArrayType | UnpackedArrayType | StructType):
            raise Exception(f"expected hw aggregate type but got {attr}")


def getBitWidth(type: Attribute) -> int:
    if isinstance(type, IntegerType):
        return type.width.data
    elif isinstance(type, ArrayType | UnpackedArrayType):
        cast_type: ArrayType[Attribute] | UnpackedArrayType[Attribute] = type
        element_width = getBitWidth(cast_type.element_type)
        if element_width < 0:
            return element_width
        assert isinstance(type.size, IntegerAttr)
        if type.size.value.data < 0:
            return -1
        return element_width * type.size.value.data
    elif isinstance(type, StructType):
        total = 0
        for field in type.fields.data:
            field_width = getBitWidth(field.field_type)
            if field_width < 0:
                return field_width
            total = total + field_width
        return total
    elif isinstance(type, EnumType):
        num = len(type.fields.data)
        if num > 1:
            return (num - 1).bit_length()
        return 1
    else:
        return -1


class KnownBitWidthType(AttrConstraint):
    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if getBitWidth(attr) < 0:
            raise VerifyException("bitwidth of type {attr} cannot be computed")


################################################################################
### Operations
################################################################################


@irdl_op_definition
class ConstantOp(IRDLOperation):
    """
    The constant operation produces a constant value of standard integer type
    without a sign.
    """

    name = "hw.constant"

    T = Annotated[IntegerType, ConstraintVar("T")]
    result: OpResult = result_def(T)
    value: Attribute = attr_def(IntegerAttr[T])

    traits = frozenset((ConstantLike(),))

    def __init__(self, value: AnyIntegerAttr):
        super().__init__(
            operands=[],
            result_types=[value.type],
            properties={},
            attributes={"value": value},
        )

    @classmethod
    def parse(cls, parser: Parser):
        attrs = parser.parse_optional_attr_dict_with_reserved_attr_names(["value"])

        p0 = parser.pos
        value = parser.parse_attribute()

        if not isa(value, AnyIntegerAttr):
            parser.raise_error("Invalid constant value", p0, parser.pos)

        c = ConstantOp(value)
        if attrs is not None:
            c.attributes.update(attrs.data)
        return c

    def print(self, printer: Printer):
        printer.print_op_attributes(self.attributes, reserved_attr_names=["value"])
        printer.print(" ")
        printer.print_attribute(self.value)


@irdl_op_definition
class AggregateConstantOp(IRDLOperation):
    """
    This operation produces a constant value of an aggregate type. Clock and
    reset values are supported. For nested aggregates, embedded arrays are used.
    """

    name = "hw.aggregate_constant"

    value: ArrayAttr[Attribute] = attr_def(ArrayAttr[Attribute])
    result: OpResult = result_def(HWAggregateType())

    traits = frozenset((ConstantLike(),))

    def __init__(self, value: Attribute, type: Attribute):
        super().__init__(
            operands=[], result_types=[type], properties={}, attributes={"value": value}
        )

    @classmethod
    def parse(cls, parser: Parser):
        value = parser.parse_attribute()
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        type = parser.parse_type()

        c = AggregateConstantOp(value, type)
        c.attributes.update(attrs)
        return c

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_attribute(self.value)
        printer.print_op_attributes(self.attributes, reserved_attr_names=["value"])
        printer.print(" : ")
        printer.print(self.result.type)

    def _check_attributes(self, type: Attribute, attribute: Attribute):
        # TODO: TypeAliasType

        if isinstance(type, StructType):
            if not isinstance(attribute, ArrayAttr):
                raise VerifyException(
                    f"expected array attribute for constant of type {type}"
                )
            attr_cast: ArrayAttr[Attribute] = attribute
            for attr, field_info in zip(attr_cast.data, type.fields.data):
                self._check_attributes(field_info.field_type, attr)
        elif isinstance(type, ArrayType | UnpackedArrayType):
            if not isinstance(attribute, ArrayAttr):
                raise VerifyException(
                    f"expected array attribute for constant of type {type}"
                )
            attr_cast: ArrayAttr[Attribute] = attribute
            for attr in attr_cast.data:
                cast_type: ArrayType[Attribute] | UnpackedArrayType[Attribute] = type
                self._check_attributes(cast_type.element_type, attr)
        elif isinstance(type, EnumType):
            if not isinstance(attribute, StringAttr):
                raise VerifyException(
                    f"expected string attribute for constant of type {type}"
                )
        elif isinstance(type, IntegerType):
            if not isinstance(attribute, IntegerAttr):
                raise VerifyException(
                    f"expected integer attribute for constant of type {type}"
                )
            int_attr: IntegerAttr[IntegerType] = attribute
            if int_attr.type.width != type.width:
                raise VerifyException(
                    "constant attribute bitwidth doesn't match return type"
                )
        else:
            raise VerifyException(f"unknown element type {type}")

    def verify_(self):
        self._check_attributes(self.result.type, self.value)


@irdl_op_definition
class BitcastOp(IRDLOperation):
    """
    Reinterpret one value to another value of the same size and potentially
    different type. See the `hw` dialect rationale document for more details.
    """

    name = "hw.bitcast"

    input: Operand = operand_def(KnownBitWidthType())
    result: OpResult = result_def(KnownBitWidthType())

    def __init__(self, input: SSAValue | Operation, result_type: Attribute):
        return super().__init__(operands=[input], result_types=[result_type])

    @classmethod
    def parse(cls, parser: Parser):
        input = parser.parse_unresolved_operand()
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        op_type = parser.parse_function_type()
        input = parser.resolve_operand(input, op_type.inputs.data[0])

        c = BitcastOp(input, op_type.outputs.data[0])
        c.attributes.update(attrs)
        return c

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_ssa_value(self.input)
        printer.print_op_attributes(self.attributes)
        printer.print(" : ")
        printer.print_operation_type(self)

    def verify_(self):
        if getBitWidth(self.input.type) != getBitWidth(self.result.type):
            raise VerifyException("bitwidth of input must match result")


@irdl_op_definition
class ArrayCreateOp(IRDLOperation):
    """
    Creates an array from a variable set of values. One or more values must be
    listed.
    """

    name = "hw.array_create"

    T = Annotated[Attribute, ConstraintVar("T")]

    inputs: VarOperand = var_operand_def(T)
    result: OpResult = result_def(ArrayType[T])

    @classmethod
    def parse(cls, parser: Parser):
        elements = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_unresolved_operand
        )
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        element_type = parser.parse_type()
        elements = parser.resolve_operands(
            elements, len(elements) * [element_type], parser.pos
        )

        c = cls.create(
            operands=elements,
            result_types=[ArrayType[Attribute](len(elements), element_type)],
        )
        c.attributes.update(attrs)
        return c

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_list(self.inputs, printer.print_ssa_value)
        printer.print_op_attributes(self.attributes)
        printer.print(" : ")
        assert isinstance(self.result.type, ArrayType)
        cast_type: ArrayType[Attribute] = self.result.type
        printer.print(cast_type.element_type)

    def verify_(self):
        assert isinstance(self.result.type, ArrayType)
        assert isinstance(self.result.type.size, IntegerAttr)
        if len(self.inputs) != self.result.type.size.value.data:
            raise VerifyException("number of inputs must match size of array type")


@irdl_op_definition
class ArrayConcatOp(IRDLOperation):
    """
    Creates an array by concatenating a variable set of arrays. One or more
    values must be listed.
    """

    name = "hw.array_concat"

    T = Annotated[Attribute, ConstraintVar("T")]

    inputs: VarOperand = var_operand_def(ArrayType[T])
    result: OpResult = result_def(ArrayType[T])

    @classmethod
    def parse(cls, parser: Parser):
        elements = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_unresolved_operand
        )
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        array_types = parser.parse_comma_separated_list(
            parser.Delimiter.NONE, parser.parse_type
        )
        elements = parser.resolve_operands(elements, array_types, parser.pos)

        assert isinstance(array_types[0], ArrayType)
        cast_type: ArrayType[Attribute] = array_types[0]
        total_num_elements = 0
        for ty in array_types:
            assert isinstance(ty, ArrayType)
            assert isinstance(ty.size, IntegerAttr)
            total_num_elements = total_num_elements + ty.size.value.data

        c = cls.create(
            operands=elements,
            result_types=[
                ArrayType[Attribute](total_num_elements, cast_type.element_type)
            ],
        )
        c.attributes.update(attrs)
        return c

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_list(self.inputs, printer.print_ssa_value)
        printer.print_op_attributes(self.attributes)
        printer.print(" : ")
        printer.print_list(self.inputs, lambda el: printer.print(el.type))

    def verify_(self):
        assert isinstance(self.result.type, ArrayType)

        total = 0
        for input in self.inputs:
            assert isinstance(input.type, ArrayType)
            assert isinstance(input.type.size, IntegerAttr)
            total = total + input.type.size.value.data

        assert isinstance(self.result.type.size, IntegerAttr)
        if total != self.result.type.size.value.data:
            raise VerifyException(
                "number of elements in the inputs must match size of array type"
            )


@irdl_op_definition
class ArraySliceOp(IRDLOperation):
    """
    Extracts a sub-range from an array. The range is from `lowIndex` to
    `lowIndex` + the number of elements in the return type, non-inclusive on the
    high end. For instance,

    ```
    // Slices 16 elements starting at '%offset'.
    %subArray = hw.slice %largerArray at %offset :
        (!hw.array<1024xi8>) -> !hw.array<16xi8>
    ```

    Would translate to the following SystemVerilog:

    ```
    logic [7:0][15:0] subArray = largerArray[offset +: 16];
    ```

    Width of 'idx' is defined to be the precise number of bits required to index
    the 'input' array. More precisely: for an input array of size M, the width
    of 'idx' is ceil(log2(M)). Lower and upper bound indexes which are larger
    than the size of the 'input' array results in undefined behavior.
    """

    name = "hw.array_slice"

    T = Annotated[Attribute, ConstraintVar("T")]

    input: Operand = operand_def(ArrayType[T])
    index: Operand = operand_def(IntegerType)
    result: OpResult = result_def(ArrayType[T])

    @classmethod
    def parse(cls, parser: Parser):
        input = parser.parse_unresolved_operand()
        parser.parse_punctuation("[")
        index = parser.parse_unresolved_operand()
        parser.parse_punctuation("]")
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        op_type = parser.parse_function_type()
        input = parser.resolve_operand(input, op_type.inputs.data[0])
        assert isinstance(input.type, ArrayType)
        assert isinstance(input.type.size, IntegerAttr)
        index = parser.resolve_operand(
            index, IntegerType((input.type.size.value.data - 1).bit_length())
        )

        return cls.create(
            operands=[input, index], attributes=attrs, result_types=op_type.outputs.data
        )

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_ssa_value(self.input)
        printer.print("[")
        printer.print_ssa_value(self.index)
        printer.print("]")
        printer.print_op_attributes(self.attributes)
        printer.print(" : (")
        printer.print(self.input.type)
        printer.print(") -> ")
        printer.print(self.result.type)

    def verify_(self):
        assert isinstance(self.input.type, ArrayType)
        assert isinstance(self.input.type.size, IntegerAttr)
        assert isinstance(self.index.type, IntegerType)
        expected_width: int = (self.input.type.size.value.data - 1).bit_length()
        width: int = self.index.type.width.data
        if expected_width == 0 and width == 1:
            return
        if expected_width != width:
            raise VerifyException(
                f"ArraySlice: index width must match clog2 of array size: expected {expected_width}, but got {width}"
            )


@irdl_op_definition
class ArrayGetOp(IRDLOperation):
    """Get the value in an array at the specified index"""

    name = "hw.array_get"

    T = Annotated[Attribute, ConstraintVar("T")]

    input: Operand = operand_def(ArrayType[T])
    index: Operand = operand_def(IntegerType)
    result: OpResult = result_def(T)

    @classmethod
    def parse(cls, parser: Parser):
        input = parser.parse_unresolved_operand()
        parser.parse_punctuation("[")
        index = parser.parse_unresolved_operand()
        parser.parse_punctuation("]")
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        input_type = parser.parse_type()
        parser.parse_punctuation(",")
        index_type = parser.parse_type()
        input = parser.resolve_operand(input, input_type)
        index = parser.resolve_operand(index, index_type)

        assert isinstance(input_type, ArrayType)
        cast_type: ArrayType[Attribute] = input_type
        c = cls.create(
            operands=[input, index],
            attributes=attrs,
            result_types=[cast_type.element_type],
        )
        return c

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_ssa_value(self.input)
        printer.print("[")
        printer.print_ssa_value(self.index)
        printer.print("]")
        printer.print_op_attributes(self.attributes)
        printer.print(" : ")
        printer.print(self.input.type)
        printer.print(", ")
        printer.print(self.index.type)

    def verify_(self):
        assert isinstance(self.input.type, ArrayType)
        assert isinstance(self.input.type.size, IntegerAttr)
        assert isinstance(self.index.type, IntegerType)

        expected_width: int = (self.input.type.size.value.data - 1).bit_length()
        width: int = self.index.type.width.data
        if expected_width == 0 and width == 1:
            return
        if expected_width != width:
            raise VerifyException(
                f"ArrayGet: index width must match clog2 of array size: expected {expected_width}, but got {width}"
            )


@irdl_op_definition
class StructCreateOp(IRDLOperation):
    """Extract a named field from a struct."""

    name = "hw.struct_create"

    fields: VarOperand = var_operand_def(Attribute)
    result: OpResult = result_def(StructType)

    @classmethod
    def parse(cls, parser: Parser):
        fields = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_unresolved_operand
        )
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        struct_type = parser.parse_type()
        assert isinstance(struct_type, StructType)
        field_types = [field.field_type for field in struct_type.fields.data]
        fields = parser.resolve_operands(fields, field_types, parser.pos)

        return cls.create(operands=fields, attributes=attrs, result_types=[struct_type])

    def print(self, printer: Printer):
        printer.print(" (")
        printer.print_list(self.fields, printer.print_ssa_value)
        printer.print(")")
        printer.print_op_attributes(self.attributes)
        printer.print(" : ")
        printer.print(self.result.type)

    def verify_(self):
        assert isinstance(self.result.type, StructType)

        if len(self.fields) != len(self.result.type.fields.data):
            raise VerifyException(
                "number of inputs must match number of fields in struct type"
            )

        for input, field in zip(self.fields, self.result.type.fields.data):
            if input.type != field.field_type:
                raise VerifyException("type of input value must match field type")


@irdl_op_definition
class StructExplodeOp(IRDLOperation):
    """Expand a struct into its constituent parts."""

    name = "hw.struct_explode"

    input: Operand = operand_def(StructType)
    result: VarOpResult = var_result_def(Attribute)

    @classmethod
    def parse(cls, parser: Parser):
        struct = parser.parse_unresolved_operand()
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        struct_type = parser.parse_type()
        assert isinstance(struct_type, StructType)
        field_types = [field.field_type for field in struct_type.fields.data]
        struct = parser.resolve_operand(struct, struct_type)

        c = cls.create(operands=[struct], result_types=field_types)
        c.attributes.update(attrs)
        return c

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_ssa_value(self.input)
        printer.print_op_attributes(self.attributes)
        printer.print(" : ")
        printer.print(self.input.type)

    def verify_(self):
        assert isinstance(self.input.type, StructType)

        if len(self.results) != len(self.input.type.fields.data):
            raise VerifyException(
                "number of inputs must match number of fields in struct type"
            )

        for result, field in zip(self.results, self.input.type.fields.data):
            if result.type != field.field_type:
                raise VerifyException("type of input value must match field type")


@irdl_op_definition
class StructExtractOp(IRDLOperation):
    """Extract a named field from a struct."""

    name = "hw.struct_extract"

    input: Operand = operand_def(StructType)
    field: StringAttr = attr_def(StringAttr)
    result: OpResult = result_def(Attribute)

    @classmethod
    def parse(cls, parser: Parser):
        struct = parser.parse_unresolved_operand()
        parser.parse_punctuation("[")
        name = parser.parse_str_literal()
        parser.parse_punctuation("]")
        attrs = parser.parse_optional_attr_dict_with_reserved_attr_names(["field"])
        parser.parse_punctuation(":")
        struct_type = parser.parse_type()
        assert isinstance(struct_type, StructType)
        struct = parser.resolve_operand(struct, struct_type)
        field_type = [
            field.field_type
            for field in struct_type.fields.data
            if field.field_name.data == name
        ]
        attributes = dict[str, Attribute]()
        if attrs is not None:
            attributes = attrs.data
        attributes["field"] = StringAttr(name)

        c = cls.create(
            operands=[struct], attributes=attributes, result_types=field_type
        )
        return c

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_ssa_value(self.input)
        printer.print("[")
        printer.print(self.field)
        printer.print("]")
        printer.print_op_attributes(self.attributes, reserved_attr_names=["field"])
        printer.print(" : ")
        printer.print(self.input.type)

    def verify_(self):
        assert isinstance(self.input.type, StructType)

        found: bool = False
        for field in self.input.type.fields.data:
            if field.field_name.data == self.field.data:
                found = True
                if field.field_type != self.result.type:
                    raise VerifyException(
                        "result must have the same type as extracted struct field"
                    )

        if not found:
            raise VerifyException(
                "field attribute must match the name of a struct field"
            )


@irdl_op_definition
class StructInjectOp(IRDLOperation):
    """Inject a value into a named field of a struct."""

    name = "hw.struct_inject"

    T = Annotated[StructType, ConstraintVar("T")]

    input: Operand = operand_def(T)
    new_value: Operand = operand_def(Attribute)
    field: StringAttr = attr_def(StringAttr)
    result: OpResult = result_def(T)

    @classmethod
    def parse(cls, parser: Parser):
        struct = parser.parse_unresolved_operand()
        parser.parse_punctuation("[")
        name = parser.parse_str_literal()
        parser.parse_punctuation("]")
        parser.parse_punctuation(",")
        new_value = parser.parse_unresolved_operand()
        attrs = parser.parse_optional_attr_dict_with_reserved_attr_names(["field"])
        parser.parse_punctuation(":")
        struct_type = parser.parse_type()
        assert isinstance(struct_type, StructType)
        field_type = [
            field.field_type
            for field in struct_type.fields.data
            if field.field_name.data == name
        ]
        struct = parser.resolve_operand(struct, struct_type)
        assert len(field_type) == 1
        new_value = parser.resolve_operand(new_value, field_type[0])
        attributes = dict[str, Attribute]()
        if attrs is not None:
            attributes = attrs.data
        attributes["field"] = StringAttr(name)

        c = cls.create(
            operands=[struct, new_value],
            attributes=attributes,
            result_types=[struct_type],
        )
        return c

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_ssa_value(self.input)
        printer.print("[")
        printer.print(self.field)
        printer.print("], ")
        printer.print_ssa_value(self.new_value)
        printer.print_op_attributes(self.attributes, reserved_attr_names=["field"])
        printer.print(" : ")
        printer.print(self.input.type)

    def verify_(self):
        assert isinstance(self.input.type, StructType)

        found: bool = False
        for field in self.input.type.fields.data:
            if field.field_name.data == self.field.data:
                found = True
                if field.field_type != self.new_value.type:
                    raise VerifyException(
                        "new value must have the same type as the struct field to inject into"
                    )

        if not found:
            raise VerifyException(
                "field attribute must match the name of a struct field"
            )


@irdl_op_definition
class OutputOp(IRDLOperation):
    """ "
    HW termination operation

    Marks the end of a region in the HW dialect and the values to put on the
    output ports.
    """

    name = "hw.output"

    arguments: VarOperand = var_operand_def(Attribute)

    traits = traits_def(
        lambda: frozenset([HasParent(HWModuleOp), IsTerminator(), Pure()])
    )

    def __init__(self, *return_vals: SSAValue | Operation):
        super().__init__(operands=[return_vals])

    def print(self, printer: Printer):
        print_return_op_like(printer, self.attributes, self.arguments)

    @classmethod
    def parse(cls, parser: Parser):
        attrs, args = parse_return_op_like(parser)
        op = OutputOp(*args)
        op.attributes.update(attrs)
        return op

    def verify_(self) -> None:
        return
        # module_op = self.parent_op()
        # assert isinstance(module_op, HWModuleOp)

        # module_output_types = module_op.module_type.outputs.data
        # output_types = tuple(arg.type for arg in self.arguments)
        # if module_output_types != output_types:
        #     raise VerifyException(
        #         "Expected operands to have the same types as the module output types"
        #     )


@irdl_op_definition
class HWModuleOp(IRDLOperation):
    """
    Represents a Verilog module, including a given name, a list of ports, a list
    of parameters, and a body that represents the connections within the module.
    """

    name = "hw.module"

    body: Region = region_def("single_block")
    sym_name: StringAttr = attr_def(StringAttr)
    module_type: ModuleType = attr_def(ModuleType)
    per_port_attrs: ArrayAttr[DictionaryAttr] | None = opt_attr_def(
        ArrayAttr[DictionaryAttr]
    )
    port_locs: ArrayAttr[LocationAttr] | None = opt_attr_def(ArrayAttr[LocationAttr])
    parameters: ArrayAttr[ParamDeclAttr] = attr_def(ArrayAttr[ParamDeclAttr])
    comment: StringAttr | None = opt_attr_def(StringAttr)

    # TODO: visibility

    traits = frozenset(
        [
            IsolatedFromAbove(),
            SymbolOpInterface(),
            InnerSymbolTable(),
            HWModuleLike(),
            PortList(),
            HasParent(ModuleOp),
            SingleBlockImplicitTerminator(OutputOp),
        ]
    )

    def __init__(
        self,
        name: StringAttr,
        region: Region | None = None,
        ports: Sequence[PortAttr] = [],
        parameters: ArrayAttr[ParamDeclAttr] = ArrayAttr[ParamDeclAttr]([]),
        attributes: Sequence[Attribute] = [],
        comment: StringAttr | None = None,
        attr_dict: DictionaryAttr | None = None,
    ):
        input_port_types: Sequence[Attribute] = []
        for port in ports:
            if port.port_dir != Direction.OUTPUT:
                input_port_types.append(port.port_type)

        if region is None:
            region = Region(Block(arg_types=input_port_types))

        # TODO: port_locs
        attrs: dict[str, Attribute] = {}
        if attr_dict is not None:
            attrs = attr_dict.data

        attrs["sym_name"] = name
        attrs["module_type"] = ModuleType(ports)
        attrs["parameters"] = parameters

        if len(attributes) != 0:
            attrs["per_port_attrs"] = ArrayAttr(attributes)
        if comment is not None:
            attrs["comment"] = comment

        super().__init__(attributes=attrs, regions=[region])

    def __post_init__(self):
        for trait in self.get_traits_of_type(SingleBlockImplicitTerminator):
            ensure_terminator(self, trait)

    def verify_(self) -> None:
        return

    @classmethod
    def parse(cls, parser: Parser):
        # Parse visibility keyword if present
        # if parser.parse_optional_keyword("public"):
        #     visibility = "public"
        # elif parser.parse_optional_keyword("nested"):
        #     visibility = "nested"
        # elif parser.parse_optional_keyword("private"):
        #     visibility = "private"
        # else:
        #     visibility = None

        name = parser.parse_symbol_name()
        # TODO: parse parameters

        def parse_port() -> tuple[PortAttr, Parser.Argument | None]:
            dir = Direction.OUTPUT
            name = ""
            type = Attribute()
            arg = None
            if parser.parse_optional_characters("inout"):
                dir = Direction.INOUT
                arg = parser.parse_argument()
                name = arg.name.text
                type = arg.type
            elif parser.parse_optional_characters("in"):
                dir = Direction.INPUT
                arg = parser.parse_argument()
                name = arg.name.text
                type = arg.type
            elif parser.parse_optional_characters("out"):
                name = parser.parse_identifier()
                parser.parse_punctuation(":")
                type = parser.parse_type()

            # TODO: explicit port names
            assert isinstance(type, Attribute)
            return PortAttr(dir, StringAttr(name), type), arg

        signature = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parse_port
        )
        if len(signature) > 0:
            (ports, args) = zip(*signature)
        else:
            (ports, args) = ([], [])
        # assert(isinstance(ports, Sequence[PortAttr]))
        # assert(isinstance(args, Sequence[Parser.Argument | None]))
        attr_dict = parser.parse_optional_attr_dict_with_keyword(
            ["parameters", "sym_name", "module_type", "per_port_attrs", "port_locs"]
        )

        inputs: Sequence[Parser.Argument] = []
        for arg in args:
            if arg is not None:
                inputs.append(arg)
        region = parser.parse_region(inputs)
        module_op = HWModuleOp(name, region=region, ports=ports, attr_dict=attr_dict)
        # region.blocks[0].last_op
        # module_op.add_region(region)
        # trait = module_op.get_trait(SingleBlockImplicitTerminator)
        # assert(isinstance(trait, SingleBlockImplicitTerminator))
        # ensure_terminator(module_op, trait)
        return module_op

    def print(self, printer: Printer):
        printer.print(f" @{self.sym_name.data}(")
        args: Sequence[BlockArgument | None] = []
        i = 0
        for port in self.module_type.ports:
            if port.port_dir.data == Direction.OUTPUT:
                args.append(None)
                continue
            args.append(self.regions[0].blocks[0].args[i])
            i = i + 1

        def printPort(portAndArg: tuple[PortAttr, BlockArgument | None]) -> None:
            port = portAndArg[0]
            arg = portAndArg[1]
            port.port_dir.print_parameter(printer)
            printer.print(" ")
            if arg is None:
                printer.print(f"{port.port_name.data}: {port.port_type}")
            else:
                printer.print(arg)
                printer.print(f": {port.port_type}")

        printer.print_list(zip(self.module_type.ports, args), printPort)
        printer.print(")")
        printer.print_op_attributes(
            self.attributes,
            reserved_attr_names=[
                "parameters",
                "sym_name",
                "module_type",
                "per_port_attrs",
                "port_locs",
            ],
            print_keyword=True,
        )

        printer.print(" ")
        printer.print_region(self.regions[0], print_entry_block_args=False)

    @property
    def is_declaration(self) -> bool:
        """
        A helper to identify functions that are external declarations (have an empty
        function body)
        """
        return False


@irdl_op_definition
class InstanceOp(IRDLOperation):
    """ "
    Create an instance of a module.

    This represents an instance of a module. The inputs and results are the
    referenced module's inputs and outputs.  The `argNames` and `resultNames`
    attributes must match the referenced module.
    """

    name = "hw.instance"

    instance_name: StringAttr = attr_def(StringAttr)
    module_name: FlatSymbolRefAttr = attr_def(FlatSymbolRefAttr)
    inputs: VarOperand = var_operand_def(Attribute)
    arg_names: ArrayAttr[StringAttr] = attr_def(ArrayAttr[StringAttr])
    result_names: ArrayAttr[StringAttr] = attr_def(ArrayAttr[StringAttr])
    parameters: ArrayAttr[ParamDeclAttr] = attr_def(ArrayAttr[ParamDeclAttr])
    inner_sym: InnerSymAttr | None = opt_attr_def(InnerSymAttr)
    results: VarOpResult = var_result_def(Attribute)

    traits = frozenset(
        [PortList(), InnerSymbol(), HWInstanceLike()]
    )  # add SymbolUser trait

    def __init__(
        self,
        module: Operation,
        name: StringAttr,
        inputs: Sequence[SSAValue],
        parameters: Sequence[ParamDeclAttr] = [],
        inner_sym: InnerSymAttr | None = None,
    ):
        super().__init__(operands=[inputs])

    @classmethod
    def parse(cls, parser: Parser):
        attrs, args = parse_return_op_like(parser)
        op = OutputOp(*args)
        op.attributes.update(attrs)
        return op

    def print(self, printer: Printer):
        print_return_op_like(printer, self.attributes, self.arguments)

    def verify_(self):
        pass


################################################################################
### Registration
################################################################################

HW = Dialect(
    [
        ConstantOp,
        AggregateConstantOp,
        HWModuleOp,
        OutputOp,
        InstanceOp,
        BitcastOp,
        StructCreateOp,
        StructExplodeOp,
        StructExtractOp,
        StructInjectOp,
        ArrayCreateOp,
        ArrayConcatOp,
        ArraySliceOp,
        ArrayGetOp,
    ],
    [
        ArrayType,
        UnpackedArrayType,
        ParamDeclAttr,
        InnerRefAttr,
        ModuleType,
        PortAttr,
        DirectionAttr,
        StructFieldAttr,
        StructType,
        EnumType,
    ],
)
