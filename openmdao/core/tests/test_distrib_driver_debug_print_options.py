import os
import sys
import unittest
import numpy as np
from six.moves import cStringIO
from six import StringIO

from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp

from openmdao.utils.mpi import MPI
from openmdao.utils.array_utils import evenly_distrib_idxs

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.utils.assert_utils import assert_rel_error

@unittest.skipIf(PETScVector is None or os.environ.get("TRAVIS"),
                 "PETSc is required." if PETScVector is None
                 else "Unreliable on Travis CI.")
class DistributedDriverDebugPrintOptionsTest(unittest.TestCase):

    N_PROCS = 2

    def test_distributed_driver_debug_print_options(self):

        from openmdao.utils.general_utils import set_pyoptsparse_opt
        from openmdao.utils.mpi import MPI

        if MPI:
            from openmdao.api import PETScVector
            vector_class = PETScVector
        else:
            PETScVector = None

        # check that pyoptsparse is installed. if it is, try to use SLSQP.
        OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')

        if OPTIMIZER:
            from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

        from openmdao.core.parallel_group import ParallelGroup
        from openmdao.components.exec_comp import ExecComp

        class Mygroup(Group):

            def setup(self):
                self.add_subsystem('indep_var_comp', IndepVarComp('x'), promotes=['*'])
                self.add_subsystem('Cy', ExecComp('y=2*x'), promotes=['*'])
                self.add_subsystem('Cc', ExecComp('c=x+2'), promotes=['*'])

                self.add_design_var('x')
                self.add_constraint('c', lower=-3.)

        prob = Problem()

        prob.model.add_subsystem('par', ParallelGroup())

        prob.model.par.add_subsystem('G1', Mygroup())
        prob.model.par.add_subsystem('G2', Mygroup())

        prob.model.add_subsystem('Obj', ExecComp('obj=y1+y2'))

        prob.model.connect('par.G1.y', 'Obj.y1')
        prob.model.connect('par.G2.y', 'Obj.y2')

        prob.model.add_objective('Obj.obj')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['print_results'] = False

        prob.driver.debug_print['debug_print'] = True
        prob.driver.debug_print['debug_print_desvars'] = True
        prob.driver.debug_print['debug_print_nl_con'] = True
        prob.driver.debug_print['debug_print_ln_con'] = True
        prob.driver.debug_print['debug_print_objective'] = True

        prob.setup(vector_class=PETScVector)

        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        sys.stdout = strout
        try:
            prob.run_driver()
        finally:
            sys.stdout = stdout

        # prob.run_driver()
        prob.cleanup()

        # stdout = sys.stdout
        # strout = StringIO()
        # sys.stdout = strout
        # try:
        #     prob.run_driver()
        # finally:
        #     sys.stdout = stdout

        output = strout.getvalue().split('\n')
        if MPI.COMM_WORLD.rank == 0:
            # Just make sure we have more than one. Not sure we will always have the same number
            #    of iterations
            self.assertTrue(output.count("Design Vars") > 1,
                            "Should be more than one design vars header printed")
            self.assertTrue(output.count("Nonlinear constraints") > 1,
                            "Should be more than one nonlinear constraint header printed")
            self.assertTrue(output.count("Linear constraints") > 1,
                            "Should be more than one linear constraint header printed")
            self.assertTrue(output.count("Objectives") > 1,
                            "Should be more than one objective header printed")

            self.assertTrue(len([s for s in output if s.startswith('par.G1.indep_var_comp.x')]) > 1,
                            "Should be more than one par.G1.indep_var_comp.x printed")
            self.assertTrue(len([s for s in output if s.startswith('par.G2.indep_var_comp.x')]) > 1,
                            "Should be more than one par.G2.indep_var_comp.x printed")
            self.assertTrue(len([s for s in output if s.startswith('par.G1.Cc.c')]) > 1,
                            "Should be more than one par.G1.Cc.c printed")
            self.assertTrue(len([s for s in output if s.startswith('par.G2.Cc.c')]) > 1,
                            "Should be more than one par.G2.Cc.c printed")
            self.assertTrue(len([s for s in output if s.startswith('None')]) > 1,
                            "Should be more than one None printed")
            self.assertTrue(len([s for s in output if s.startswith('Obj.obj')]) > 1,
                            "Should be more than one Obj.obj printed")
            # self.assertEqual(len([s for s in output if s.startswith('par.G1.indep_var_comp.x')]), 11)
            # self.assertEqual(len([s for s in output if s.startswith('par.G2.indep_var_comp.x')]), 11)
            # self.assertEqual(len([s for s in output if s.startswith('par.G1.Cc.c')]), 11)
            # self.assertEqual(len([s for s in output if s.startswith('par.G2.Cc.c')]), 11)
            # self.assertEqual(len([s for s in output if s.startswith('None')]), 11)
            # self.assertEqual(len([s for s in output if s.startswith('Obj.obj')]), 11)
        else:
            self.assertEqual(output, [''])


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()