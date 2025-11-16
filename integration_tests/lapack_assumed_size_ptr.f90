subroutine CALLEE(SEED)
      integer SEED(*)
      SEED(1) = SEED(1) + 1
      return
      end

      subroutine CALLER(SEED)
      integer SEED(4)
      call CALLEE(SEED)
      return
      end

      program LEGACY_ASSUME
      integer SEED(4)
      SEED(1) = 1
      SEED(2) = 2
      SEED(3) = 3
      SEED(4) = 4
      call CALLER(SEED)
      if (SEED(1) .ne. 2) then
         stop 1
      endif
      print *, 'legacy assumed-size passthrough ok'
      end
