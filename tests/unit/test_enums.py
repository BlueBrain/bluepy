import bluepy.enums as test_module


class Foo(test_module.OrderedEnum):
    A = 'y'
    B = 'x'


def test_ordered_enum():
    assert Foo.A > Foo.B

    assert Foo.A >= Foo.A
    assert Foo.A >= Foo.B

    assert Foo.B < Foo.A

    assert Foo.B <= Foo.A
    assert Foo.B <= Foo.B

    assert Foo.A.__ge__('y') == NotImplemented
    assert Foo.A.__gt__('y') == NotImplemented
    assert Foo.A.__lt__('y') == NotImplemented
    assert Foo.A.__le__('y') == NotImplemented
