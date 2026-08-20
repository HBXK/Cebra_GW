def test_public_imports():
    from swirl.cebra_dim_reduction import (
        CEBRAAnalysis,
        CEBRAUtils,
    )
    from swirl.stratified_gw import (

        run_analysis,
        split_point_clouds,
    )

    assert CEBRAAnalysis is not None
    assert CEBRAUtils is not None
    assert callable(run_analysis)
    assert callable(split_point_clouds)